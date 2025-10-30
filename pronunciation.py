from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io
import traceback
from pydub import AudioSegment
import re
import Levenshtein  
app = Flask(__name__)
CORS(app)

# --- NEW MODEL AND PROCESSOR ---
MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"

print(f"Loading model {MODEL_ID}...")

try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# --- Helper Function to load audio ---
def pydub_to_np(audio_segment: AudioSegment):
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    
    if audio_segment.sample_width == 2:
        samples /= np.iinfo(np.int16).max
    elif audio_segment.sample_width == 4:
        samples /= np.iinfo(np.int32).max
    elif audio_segment.sample_width == 1:
        samples = (samples - 128) / 128.0
        
    return samples, 16000


class PronunciationScorer:
    """
    Implements per-phoneme Goodness of Pronunciation (GOP) scoring
    with post-transcription alignment.
    """
    
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def get_target_phonemes(self, text):
        """
        Generates the target phoneme list from text, structured by word.
        e.g., "hello world" -> [['h', 'ə', 'l', 'oʊ'], ['w', 'ɜː', 'l', 'd']]
        """
        # Split text into words, removing punctuation but keeping apostrophes
        words = re.findall(r"\b[\w']+\b", text.lower())
        
        all_words_phonemes = []
        
        for word in words:
            if not word:
                continue
                
            # Process each word individually
            with self.processor.as_target_processor():
                phonemized = self.processor(word).input_ids
            
            phoneme_str = self.processor.decode(phonemized).lower()
            
            # Extract phonemes for this word
            word_phonemes = re.findall(r"[\w'æəɑɔɛiɪɵuʊʌʒŋʃθð]+", phoneme_str)
            
            # Add to the main list
            if word_phonemes: # Avoid adding empty lists
                all_words_phonemes.append(word_phonemes)
                
        print("phonemes (by word): ", all_words_phonemes)
        return all_words_phonemes

    def compute_gop_scores(self, test_audio, sr=16000):
        """
        Transcribes audio and returns a list of detected
        phonemes, their top-1 confidence, and their top-3 scores.
        """
        input_values = self.processor(test_audio, sampling_rate=sr, return_tensors="pt").input_values

        with torch.no_grad():
            logits = self.model(input_values).logits

        # Get probabilities for the sequence (shape: [sequence_length, vocab_size])
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get top-1 prediction (for transcription)
        confidences, predicted_ids = torch.max(probs, dim=-1)

        phoneme_scores = []
        current_phoneme_ids = []
        current_phoneme_confs = []
        current_phoneme_all_probs = [] # NEW: To store probs for averaging
        blank_token_id = self.processor.tokenizer.pad_token_id

        for i, token_id in enumerate(predicted_ids):
            token_id = token_id.item()
            conf = confidences[i].item()
            all_probs_for_this_step = probs[i] # Get all probs for this step

            if token_id == blank_token_id:
                if current_phoneme_ids:
                    phoneme = self.processor.decode(current_phoneme_ids)
                    score = np.mean(current_phoneme_confs)
                    
                    # --- NEW: Get Top-3 for the segment ---
                    # Average the probabilities across all time-steps for this phoneme
                    avg_probs_for_segment = torch.stack(current_phoneme_all_probs).mean(dim=0)
                    top_3_seg_probs, top_3_seg_ids = torch.topk(avg_probs_for_segment, 3)
                    
                    top_3_list = []
                    for k in range(3):
                        ph_id = top_3_seg_ids[k].item()
                        ph_prob = top_3_seg_probs[k].item()
                        ph_str = self.processor.decode([ph_id]).lower().strip()
                        
                        if len(ph_str) > 0 and ' ' not in ph_str and '[' not in ph_str:
                            top_3_list.append({'phoneme': ph_str, 'score': ph_prob * 100})
                    # --- END NEW ---

                    phoneme_scores.append({'phoneme': phoneme, 'score': score * 100, 'top_3': top_3_list}) # MODIFIED
                    
                    current_phoneme_ids = []
                    current_phoneme_confs = []
                    current_phoneme_all_probs = [] # NEW
                continue

            if not current_phoneme_ids or token_id == current_phoneme_ids[-1]:
                current_phoneme_ids.append(token_id)
                current_phoneme_confs.append(conf)
                current_phoneme_all_probs.append(all_probs_for_this_step) # NEW
            else:
                phoneme = self.processor.decode(current_phoneme_ids)
                score = np.mean(current_phoneme_confs)

                # --- NEW: Get Top-3 for the segment ---
                avg_probs_for_segment = torch.stack(current_phoneme_all_probs).mean(dim=0)
                top_3_seg_probs, top_3_seg_ids = torch.topk(avg_probs_for_segment, 3)
                
                top_3_list = []
                for k in range(3):
                    ph_id = top_3_seg_ids[k].item()
                    ph_prob = top_3_seg_probs[k].item()
                    ph_str = self.processor.decode([ph_id]).lower().strip()
                    
                    if len(ph_str) > 0 and ' ' not in ph_str and '[' not in ph_str:
                        top_3_list.append({'phoneme': ph_str, 'score': ph_prob * 100})
                # --- END NEW ---
                
                phoneme_scores.append({'phoneme': phoneme, 'score': score * 100, 'top_3': top_3_list}) # MODIFIED
                
                current_phoneme_ids = [token_id]
                current_phoneme_confs = [conf]
                current_phoneme_all_probs = [all_probs_for_this_step] # NEW
        
        if current_phoneme_ids:
            phoneme = self.processor.decode(current_phoneme_ids)
            score = np.mean(current_phoneme_confs)
            
            # --- NEW: Get Top-3 for the segment ---
            avg_probs_for_segment = torch.stack(current_phoneme_all_probs).mean(dim=0)
            top_3_seg_probs, top_3_seg_ids = torch.topk(avg_probs_for_segment, 3)
            
            top_3_list = []
            for k in range(3):
                ph_id = top_3_seg_ids[k].item()
                ph_prob = top_3_seg_probs[k].item()
                ph_str = self.processor.decode([ph_id]).lower().strip()
                
                if len(ph_str) > 0 and ' ' not in ph_str and '[' not in ph_str:
                    top_3_list.append({'phoneme': ph_str, 'score': ph_prob * 100})
            # --- END NEW ---

            phoneme_scores.append({'phoneme': phoneme, 'score': score * 100, 'top_3': top_3_list}) # MODIFIED

        final_scores = []
        full_transcription = ""
        for item in phoneme_scores:
            phoneme = item['phoneme'].lower().strip()
            if len(phoneme) > 0 and ' ' not in phoneme and '[' not in phoneme:
                # Pass the full item (with top_3) to final_scores
                final_scores.append(item) 
                full_transcription += phoneme + " "

        return final_scores, full_transcription.strip()

    # <-- 2. ADD NEW ALIGNMENT FUNCTION -->
    def align_and_score(self, target_phonemes_by_word, target_phonemes, gop_scores):
        """
        Aligns target phonemes with actual phonemes, generates detailed
        per-phoneme scores, and calculates a per-word score.
        """
        # Get just the list of actual phonemes
        actual_phonemes = [item['phoneme'] for item in gop_scores]
        
        # Use Levenshtein opcodes on the flat lists
        opcodes = Levenshtein.opcodes(target_phonemes, actual_phonemes)
        
        aligned_scores = []
        total_score = 0
        score_count = 0
        
        gop_index = 0
        
        # --- NEW: Word-level scoring ---
        word_scores = []
        current_word_score_total = 0
        current_word_phoneme_count = 0
        target_phoneme_index = 0 # Tracks overall position in the flat target list
        
        # Get list of phoneme counts for each word
        word_lengths = [len(w) for w in target_phonemes_by_word]
        if not word_lengths:
            # Handle empty target text
            return [], 0.0, []
            
        current_word_boundary = word_lengths[0]
        current_word_index = 0
        # --- END NEW ---
        
        for op in opcodes:
            op_type, t_start, t_end, a_start, a_end = op
            
            # --- NEW: Helper function to check for word boundary ---
            def check_word_boundary():
                nonlocal target_phoneme_index, current_word_boundary, current_word_index
                nonlocal current_word_score_total, current_word_phoneme_count, word_scores, word_lengths
                
                # Check if this target phoneme is the last one of the current word
                if target_phoneme_index == current_word_boundary:
                    avg_score = (current_word_score_total / current_word_phoneme_count) if current_word_phoneme_count > 0 else 0.0
                    word_scores.append(avg_score)
                    
                    # Reset for next word
                    current_word_score_total = 0
                    current_word_phoneme_count = 0
                    current_word_index += 1
                    
                    # Set new boundary if there are more words
                    if current_word_index < len(word_lengths):
                        current_word_boundary += word_lengths[current_word_index]
            # --- END NEW HELPER ---

            
            if op_type == 'equal':
                for i in range(t_end - t_start):
                    target_ph = target_phonemes[t_start + i]
                    actual_item = gop_scores[gop_index]
                    aligned_scores.append({
                        'type': 'match',
                        'target': target_ph,
                        'actual': actual_item['phoneme'],
                        'score': actual_item['score']
                    })
                    total_score += actual_item['score']
                    score_count += 1
                    gop_index += 1
                    
                    # --- NEW: Word score logic ---
                    current_word_score_total += actual_item['score']
                    current_word_phoneme_count += 1
                    target_phoneme_index += 1
                    check_word_boundary()
                    # --- END NEW ---
                    
            elif op_type == 'replace':
                for i in range(t_end - t_start):
                    target_ph = target_phonemes[t_start + i]
                    phoneme_score_to_add = 0.0 # Default score for a mismatch
                    
                    if gop_index < len(gop_scores):
                        actual_item = gop_scores[gop_index]
                        
                        is_forgiven = False
                        forgiven_score = 0.0
                        
                        for top_ph in actual_item.get('top_3', []):
                            if top_ph['phoneme'] == target_ph:
                                is_forgiven = True
                                forgiven_score = top_ph['score']
                                break

                        if is_forgiven:
                            phoneme_score_to_add = forgiven_score # Use the forgiven score
                            aligned_scores.append({
                                'type': 'match',
                                'target': target_ph,
                                'actual': actual_item['phoneme'],
                                'score': forgiven_score,
                                'note': f"Forgiven mismatch (said '{actual_item['phoneme']}')"
                            })
                            total_score += forgiven_score
                        else:
                            # phoneme_score_to_add remains 0.0
                            aligned_scores.append({
                                'type': 'replace',
                                'target': target_ph,
                                'actual': actual_item['phoneme'],
                                'score': 0.0,
                                'note': f"Said '{actual_item['phoneme']}'"
                            })
                        
                        gop_index += 1
                    else:
                        aligned_scores.append({
                            'type': 'delete',
                            'target': target_ph,
                            'actual': None,
                            'score': 0.0
                        })
                    
                    score_count += 1
                    
                    # --- NEW: Word score logic ---
                    current_word_score_total += phoneme_score_to_add
                    current_word_phoneme_count += 1
                    target_phoneme_index += 1
                    check_word_boundary()
                    # --- END NEW ---
            
            elif op_type == 'delete':
                for i in range(t_end - t_start):
                    target_ph = target_phonemes[t_start + i]
                    aligned_scores.append({
                        'type': 'delete',
                        'target': target_ph,
                        'actual': None,
                        'score': 0.0
                    })
                    score_count += 1
                    
                    # --- NEW: Word score logic ---
                    current_word_score_total += 0.0 # Score is 0
                    current_word_phoneme_count += 1
                    target_phoneme_index += 1
                    check_word_boundary()
                    # --- END NEW ---
            
            elif op_type == 'insert':
                for i in range(a_end - a_start):
                    actual_item = gop_scores[gop_index]
                    aligned_scores.append({
                        'type': 'insert',
                        'target': None,
                        'actual': actual_item['phoneme'],
                        'score': actual_item['score']
                    })
                    gop_index += 1
                    # --- NEW: Insertions do not affect word score or target index ---
        
        final_total_score = (total_score / score_count) if score_count > 0 else 0.0
        
        # --- MODIFIED RETURN VALUE ---
        return aligned_scores, final_total_score, word_scores


# Initialize the scorer
scorer = PronunciationScorer(processor=processor, model=model)


@app.route('/api/score_pronunciation', methods=['POST'])
def score_pronunciation():
    """
    Endpoint: Score pronunciation accuracy
    This now returns per-phoneme scores.
    """
    try:
        audio_file = request.files.get('audio')
        target_text = request.form.get('text', '')
        
        if not audio_file or not target_text:
            return jsonify({'error': 'Audio file and target text required'}), 400
        
        # Load test audio (user's file)
        audio_bytes = audio_file.read()

        if not audio_bytes or len(audio_bytes) < 100: # Check for empty or tiny file
            return jsonify({'error': 'Audio file is empty or too short. Please record for at least 1 second.'}), 400
        
        # --- ADD CHECK FOR EMPTY FILE ---
        if not audio_bytes or len(audio_bytes) < 100:
            return jsonify({'error': 'Audio file is empty or too short.'}), 400
            
        test_audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        test_audio, sr = pydub_to_np(test_audio_segment) # sr is 16000
        
        # 1. Get per-phoneme scores (GOP)
        gop_scores, transcription = scorer.compute_gop_scores(test_audio, sr)
        
        # 2. Get target phonemes (structured by word)
        target_phonemes_by_word = scorer.get_target_phonemes(target_text)
        
        # 3. Get the list of plain words
        words = re.findall(r"\b[\w']+\b", target_text.lower())
        
        # 4. Create a flat list of target phonemes for alignment
        target_phonemes_flat = [phoneme for word_list in target_phonemes_by_word for phoneme in word_list]

        # 5. ALIGN THE SCORES (now returns word scores)
        aligned_scores, total_score, word_scores_list = scorer.align_and_score(
            target_phonemes_by_word, 
            target_phonemes_flat, 
            gop_scores
        )
        
        # 6. Combine words with their scores
        word_level_scores = []
        for i, word in enumerate(words):
            score = word_scores_list[i] if i < len(word_scores_list) else 0.0
            word_level_scores.append({'word': word, 'score': float(score)})

        
        if not gop_scores:
             return jsonify({
                'error': 'Could not detect any speech in the audio.',
                'target_text': target_text,
                'target_phonemes': target_phonemes_by_word,
                'aligned_scores': [],
                'word_level_scores': [], # Add empty list
                'transcription': '',
                'total_score': 0
             })
        
        # Provide feedback
        if total_score >= 85:
            feedback = "Excellent pronunciation!"
        elif total_score >= 70:
            feedback = "Good pronunciation with minor issues"
        elif total_score >= 50:
            feedback = "Fair pronunciation, needs improvement"
        else:
            feedback = "Pronunciation needs significant work"
        
        results = {
            'total_score': float(total_score),
            'word_level_scores': word_level_scores, # <-- ADD THIS LINE
            'target_text': target_text,
            'target_phonemes': target_phonemes_by_word, # Return the new word-structured list
            'aligned_scores': aligned_scores, # This is the new key data
            'gop_scores': gop_scores, # Keep this for debugging
            'transcription': transcription,
            'feedback': feedback
        }
        
        return jsonify(results)
    
    except Exception as e:
        print("\n" + "="*20 + " ERROR IN /api/score_pronunciation " + "="*20)
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({'error': str(e)}), 500

# --- OTHER ENDPOINTS (Add back if needed) ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': MODEL_ID})

@app.route('/')
def serve_root():
    """Serve the index.html frontend."""
    try:
        return send_from_directory('.', 'index.html')
    except FileNotFoundError:
        return "<h1>Error</h1><p>index.html not found. Please create it in the same directory.</p>", 404
    
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Pronunciation Scoring API Server")
    print("*** NEW: Running with Per-Phoneme GOP & Alignment ***")
    print(f"*** Model: {MODEL_ID} ***")
    print("="*60)
    print("\nAvailable endpoints:")
    print("  POST /api/score_pronunciation - Full pronunciation scoring")
    print("  GET  /api/health - Health check")
    print("\nThis server now has CORS enabled.")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)