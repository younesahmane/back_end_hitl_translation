import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import langdetect
import torch
from sacrebleu.metrics import CHRF
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from jsonschema import validate, ValidationError

class PreprocessorAgent:
    """
    Normalizes and structures human edits for translation corrections.
    Specialized for English/French → Arabic translation.
    """
    
    def __init__(self, source_languages: List[str] = None, target_language: str = "ar"):
        """
        Args:
            source_languages: List of ISO language codes for source languages (e.g., ['en', 'fr'])
            target_language: ISO language code for target language (default: 'ar' for Arabic)
        """
        self.source_languages = source_languages or ["en", "fr"]
        self.target_language = target_language
    
    def normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text: remove diacritics, standardize characters and punctuation.
        """
        # Normalize Arabic letters
        arabic_replacements = {
            "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
            "ى": "ي", "ئ": "ي"
        }
        for orig, repl in arabic_replacements.items():
            text = text.replace(orig, repl)

        # Normalize Arabic punctuation
        arabic_punct = {"،": ",", "؛": ";", "؟": "?", "«": '"', "»": '"', "٫": ".", "．": "."}
        for orig, repl in arabic_punct.items():
            text = text.replace(orig, repl)

        # Normalize Arabic digits to Western digits
        arabic_digits = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
                         "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"}
        for orig, repl in arabic_digits.items():
            text = text.replace(orig, repl)

        # Remove emojis and symbols
        text = re.sub(r'[\U0001F600-\U0001F64F'
                      r'\U0001F300-\U0001F5FF'
                      r'\U0001F680-\U0001F6FF]', '', text)

        # Remove Arabic diacritics (tashkeel)
        arabic_diacritics = re.compile("""
                                 ّ | َ | ً | ُ | ٌ | ِ | ٍ | ْ | ـ
                             """, re.VERBOSE)
        text = re.sub(arabic_diacritics, '', text)

        # Normalize hyphens
        hyphens = ["–", "—", "ـ", "−", "_", "\u2011"]
        for h in hyphens:
            text = text.replace(h, "-")
        text = re.sub(r'-+', '-', text)

        # normalize quotes
        text = re.sub(r'[«»“”„"]', '"', text)

        return text

    def normalize_latin(self, text: str) -> str:
        """
        Normalize Latin text (French/English): standardize quotes, apostrophes, and punctuation.
        """
        # Decode escaped quotes/backslashes
        text = text.replace('\\"', '"').replace('\\\\', '\\')

        # Unicode normalization
        text = unicodedata.normalize("NFC", text)

        # Normalize apostrophes
        apostrophes = ["'", "`", "´", "ʼ"]
        for a in apostrophes:
            text = text.replace(a, "'")

        # Normalize quotation marks
        quote_map = {
            "«": '"', "»": '"',
            """: '"', """: '"', "„": '"'
        }
        for q, repl in quote_map.items():
            text = text.replace(q, repl)

        # Normalize hyphens/dashes
        hyphens = ["–", "—", "−", "-", "‒", "_", "\u2011"]
        for h in hyphens:
            text = text.replace(h, "-")
        text = re.sub(r'-+', '-', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove control characters
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != "C")

        return text

    def normalize_whitespace_and_punctuation(self, text: str) -> str:
        """
        Normalize whitespace and remove duplicated punctuation.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove duplicated punctuation
        text = re.sub(r'([?.!,;:])\1+', r'\1', text)
        
        return text

    def normalize_text(self, text: str, is_arabic: bool = False) -> str:
        """
        Apply appropriate normalization based on language.
        
        Args:
            text: Text to normalize
            is_arabic: True if text is in Arabic, False for Latin scripts
            
        Returns:
            Normalized text
        """
        if is_arabic:
            text = self.normalize_arabic(text)
        else:
            text = self.normalize_latin(text)
        
        # Apply common normalization
        text = self.normalize_whitespace_and_punctuation(text)
        
        return text
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        """
        try:
            return langdetect.detect(text)
        except:
            return "unknown"
    
    def process(self, source: str, llm_outputs: Dict[str, str], 
                human_edit: str) -> Optional[Dict]:
        """        
        Args:
            source: Original source text (French/English) - already normalized, passed as-is
            llm_outputs: Dictionary of LLM translations in Arabic {"model1": "arabic_translation1", ...}
            human_edit: Human-corrected translation (Arabic)
            
        Returns:
            Structured JSON object or None if validation fails
        """
        
        # Source is already normalized (French/English), use as-is
        normalized_source = source
        
        # Normalize human edit (Arabic)
        normalized_edit = self.normalize_text(human_edit, is_arabic=True)
        
        # Normalize LLM outputs (Arabic)
        normalized_llm_outputs = {}
        for model, translation in llm_outputs.items():
            normalized_llm_outputs[model] = self.normalize_text(translation, is_arabic=True)
        
        # Validate human edit is not empty
        if not normalized_edit or normalized_edit.isspace():
            print("Preprocessor: Human edit is empty after normalization")
            return None
        
        # Detect language of human edit (should be Arabic)
        detected_edit_lang = self.detect_language(normalized_edit)
        
        if detected_edit_lang != self.target_language:
            print(f" Warning: Human edit language '{detected_edit_lang}' differs from expected '{self.target_language}'")
            # Continue anyway but flag it
        
        # Build structured output
        result = {
            "source": normalized_source,
            "source_language": langdetect.detect(normalized_source),  
            "llm_outputs": normalized_llm_outputs,
            "human_edit": normalized_edit,
            "target_language": self.target_language,  # Target is Arabic
            "processed_at": datetime.now().isoformat()
        }
        
        print(f"Preprocessor: Successfully processed correction")
        
        return result

class ValidatorAgent:
    """
    Translation correction validator with clear decision tree.

    Decision Flow:
    1. Check for exact duplication (>0.995) → REJECT
    2. Check for semantic errors via LLM → REJECT if meaning wrong
    3. Check for near-duplication (>0.92) → Check quality
    4. Check quality improvement → REJECT if not better
    5. Compute confidence score → ACCEPT/REJECT
    """

    def __init__(
        self,
        accept_threshold: float = 0.70,
        exact_threshold: float = 0.995,
        near_threshold: float = 0.92,
        weights: dict | None = None,
        judge_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_path: str = "./models",
        device: str | None = None,
    ):
        self.accept_threshold = accept_threshold
        self.exact_threshold = exact_threshold
        self.near_threshold = near_threshold
    
        self.weights = weights or {
            "chrf": 0.4,
            "length": 0.1,
            "llm_judge": 0.5,
        }
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chrf_metric = CHRF()
    
        # Create models directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # -----------------------------
        # Load embedding model
        # -----------------------------
        embedding_dir = os.path.join(model_path, embedding_model.replace("/", "__"))
        
        if os.path.exists(embedding_dir):
            print(f"✅ Loading embedding model from local cache: {embedding_dir}")
            try:
                self.embedder = SentenceTransformer(embedding_dir, device=self.device)
                print("✅ Embedding model loaded successfully from local cache")
            except Exception as e:
                print(f"❌ Error loading local embedding model: {e}")
                raise RuntimeError(f"Failed to load embedding model from {embedding_dir}.")
        else:
            print(f"⚠️ Local embedding model not found. Downloading from Hugging Face...")
            self.embedder = SentenceTransformer(embedding_model, device=self.device)
            os.makedirs(embedding_dir, exist_ok=True)
            self.embedder.save(embedding_dir)
            print(f"✅ Model saved to local cache: {embedding_dir}")
        
        # -----------------------------
        # Load judge model - UPDATED FOR .safetensors FILES
        # -----------------------------
        judge_dir = os.path.join(model_path, judge_model.replace("/", "__"))
        
        if not os.path.exists(judge_dir):
            print(f"⚠️ Local judge model not found. Downloading from Hugging Face...")
            self.tokenizer = AutoTokenizer.from_pretrained(judge_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                judge_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            os.makedirs(judge_dir, exist_ok=True)
            self.model.save_pretrained(judge_dir)
            self.tokenizer.save_pretrained(judge_dir)
            print(f"✅ Judge model saved to local cache: {judge_dir}")

                
        
        files = os.listdir(judge_dir)
        
        # Check for model files - accept either .safetensors or .bin
        has_safetensors = any(f.endswith('.safetensors') for f in files)
        has_pytorch_bin = any(f.endswith('.bin') for f in files)
        
        
        # Check for essential files
        essential_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        missing_essential = []
        
        for file in essential_files:
            if not os.path.exists(os.path.join(judge_dir, file)):
                missing_essential.append(file)
        
        if missing_essential:
            print(f"❌ Missing essential files: {missing_essential}")
            raise RuntimeError(f"Missing essential files: {missing_essential}")
        
        print("✅ All essential files present")
        
        # Now try to load the model
        print("\n🔄 Loading judge model (offline mode)...")
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                judge_dir,
                local_files_only=True,
                trust_remote_code=False  # Usually safe to set as False for Qwen
            )
            
            # Determine dtype
            model_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load model - transformers will automatically use .safetensors if available
            self.model = AutoModelForCausalLM.from_pretrained(
                judge_dir,
                torch_dtype=model_dtype,
                local_files_only=True,
                trust_remote_code=False
            ).to(self.device)
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            
            # Try with trust_remote_code=True if the first attempt failed
            print("🔄 Trying with trust_remote_code=True...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    judge_dir,
                    local_files_only=True,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    judge_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    local_files_only=True,
                    trust_remote_code=True
                ).to(self.device)
                print("✅ Model loaded with trust_remote_code=True!")
            except Exception as e2:
                print(f"❌ Still failed: {e2}")
                
                # Last resort: create a symbolic link if .safetensors exists but code expects .bin
                if has_safetensors and not has_pytorch_bin:
                    print("🔄 Creating symbolic link: model.safetensors → pytorch_model.bin")
                    try:
                        import sys
                        if sys.platform == "win32":
                            # Windows
                            import subprocess
                            safetensors_path = os.path.join(judge_dir, "model.safetensors")
                            bin_path = os.path.join(judge_dir, "pytorch_model.bin")
                            subprocess.run(['mklink', bin_path, safetensors_path], shell=True)
                        else:
                            # Unix/Linux/Mac
                            os.symlink(
                                os.path.join(judge_dir, "model.safetensors"),
                                os.path.join(judge_dir, "pytorch_model.bin")
                            )
                        
                        # Try loading again
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            judge_dir,
                            local_files_only=True
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            judge_dir,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                            local_files_only=True
                        ).to(self.device)
                    except Exception as e3:
                        raise RuntimeError(f"Failed to load model from {judge_dir}")
                else:
                    raise RuntimeError(f"Failed to load model from {judge_dir}")
        
        self.model.eval()

    
    # -------------------------
    # Metric helpers
    # -------------------------
    def _embedding_similarity(self, a: str, b: str) -> float:
        """Compute cosine similarity between two texts."""
        ea = self.embedder.encode(a, normalize_embeddings=True)
        eb = self.embedder.encode(b, normalize_embeddings=True)
        return float(util.cos_sim(ea, eb))

    def _chrf(self, ref: str, hyp: str) -> float:
        return self.chrf_metric.sentence_score(hyp, [ref]).score / 100.0

    def _length_ratio(self, ref: str, hyp: str) -> float:
        return min(len(ref), len(hyp)) / max(len(ref), len(hyp), 1)

    # -------------------------
    # LLM judges - FIXED VERSION
    # -------------------------
    def _llm_judge_meaning(self, source: str, human: str) -> float:
        """
        Verify if translation preserves source meaning.
        """
        prompt = (
            "You are a translation accuracy checker. "
            "Does the Arabic translation preserve the EXACT meaning of the English?\n\n"
            f"English: {source}\n"
            f"Arabic: {human}\n\n"
            "Check these specific things:\n"
            "1. Are key nouns the same? (e.g., if English says 'apples', Arabic should say 'تفاح' not 'موز')\n"
            "2. Is the action/verb the same?\n"
            "3. Are quantities/numbers the same?\n"
            "4. Is the time reference the same?\n"
            "5. Is the meaning preserved (not opposite)?\n\n"
            "Answer with ONLY '1' if ALL meaning is preserved correctly, or '0' if ANY part is wrong.\n"
        )
        result = self._run_judge_simple(prompt)
        return result

    def _llm_judge_quality(self, source: str, llm_outputs: Dict[str, str], human: str) -> float:
        """
        Judge if human translation is better than model outputs.
        """
        # First, compute all similarities
        similarities = []
        for model_output in llm_outputs.values():
            sim = self._embedding_similarity(human, model_output)
            similarities.append(sim)
        
        max_sim = max(similarities) if similarities else 0
        
        prompt = (
            "Compare these translations and decide if the human one is BETTER.\n\n"
            f"Source English: {source}\n\n"
            "Machine translations:\n"
        )
        
        for i, (model_name, translation) in enumerate(llm_outputs.items(), 1):
            prompt += f"Machine {i}: {translation}\n"
        
        prompt += f"\nHuman translation: {human}\n\n"
        
        # Add strictness warning for high similarity
        if max_sim > 0.85:
            prompt += (
                "NOTE: The human translation is very similar to machine ones. "
                "Only mark as better if it fixes clear errors or is significantly more natural.\n\n"
            )
        
        prompt += (
            "Consider:\n"
            "1. Grammar correctness\n"
            "2. Natural Arabic phrasing\n"
            "3. Appropriate word choice\n"
            "4. Overall fluency\n\n"
            "Answer with '1' if human is CLEARLY better, '0' if equal or worse.\n"
            "Only '1' or '0'.\n"
        )
        
        return self._run_judge_simple(prompt)

    def _run_judge_simple(self, prompt: str) -> float:
        """Simplified judge runner."""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024  # Increased length
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.1,
            )
        
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        
        # Look for 1 or 0 in the output
        if "1" in decoded and "0" not in decoded[-10:]:  # Check last part
            return 1.0
        elif "0" in decoded and "1" not in decoded[-10:]:
            return 0.0
        else:
            # Default to 0 if unclear
            return 0.0

    # -------------------------
    # Quality improvement check
    # -------------------------
    def _check_quality_improvement(self, human: str, llm_outputs: Dict[str, str]) -> float:
        """
        Check if human translation shows quality improvement.
        Returns score from 0-1.
        """
        # 1. Check if human is longer than all models (often indicates improvement)
        human_len = len(human)
        max_model_len = max(len(t) for t in llm_outputs.values())
        
        if human_len > max_model_len * 1.3:  # 30% longer
            length_score = 0.7
        elif human_len > max_model_len:
            length_score = 0.5
        else:
            length_score = 0.2
        
        # 2. Check CHRF improvement
        chrf_scores = []
        for model_output in llm_outputs.values():
            # Human compared to model (reverse of usual)
            score = self._chrf(model_output, human)
            chrf_scores.append(score)
        
        avg_chrf = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0
        chrf_score = min(avg_chrf * 2, 1.0)  # Scale up
        
        # 3. Combine scores
        return (length_score * 0.3 + chrf_score * 0.7)

    
    def validate(
        self,
        source_text: str,
        language_pair: Tuple[str, str],
        llm_outputs: Dict[str, str],
        human_translation: str,
    ) -> Dict:
        """
        Validates a human translation against LLM outputs.
        """
        # Get similarity scores
        sim_to_models = [
            self._embedding_similarity(human_translation, t)
            for t in llm_outputs.values()
        ]
        max_sim = max(sim_to_models) if sim_to_models else 0

        # -----------------------------
        # STEP 1: Check for exact duplication
        # -----------------------------
        if max_sim >= self.exact_threshold:
            return {
                "source_text": source_text,
                "language-pair": f"{language_pair[0]}-{language_pair[1]}",
                "llm_outputs": llm_outputs,
                "human_translation": human_translation,
                "score": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decision": "reject",
                "reason": f"human edit too similar to model output (similarity={max_sim:.3f})",
            }

        # -----------------------------
        # STEP 2: Check semantic correctness
        # -----------------------------
        meaning_preserved = self._llm_judge_meaning(source_text, human_translation)
        
        if meaning_preserved < 0.5:  
            return {
                "source_text": source_text,
                "language-pair": f"{language_pair[0]}-{language_pair[1]}",
                "llm_outputs": llm_outputs,
                "human_translation": human_translation,
                "score": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decision": "reject",
                "reason": "translation may not preserve source meaning",
            }

        # -----------------------------
        # STEP 3: Check for near-duplication and quality
        # -----------------------------
        if max_sim >= self.near_threshold:
            # For high similarity, we need to check quality
            is_better = self._llm_judge_quality(source_text, llm_outputs, human_translation)
            
            if is_better < 0.5:  # Not clearly better
                # Additional check: maybe it's still good enough
                quality_score = self._check_quality_improvement(human_translation, llm_outputs)
                if quality_score < 0.6:  # Not enough improvement
                    return {
                        "source_text": source_text,
                        "language-pair": f"{language_pair[0]}-{language_pair[1]}",
                        "llm_outputs": llm_outputs,
                        "human_translation": human_translation,
                        "score": quality_score,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "decision": "reject",
                        "reason": f"high similarity ({max_sim:.3f}) with insufficient improvement",
                    }
                else:
                    # Has some improvement, continue to scoring
                    is_better = 1.0
        else:
            # Lower similarity, just check if it's better
            is_better = self._llm_judge_quality(source_text, llm_outputs, human_translation)
            
            if is_better < 0.5:
                quality_score = self._check_quality_improvement(human_translation, llm_outputs)
                if quality_score < 0.5:
                    return {
                        "source_text": source_text,
                        "language-pair": f"{language_pair[0]}-{language_pair[1]}",
                        "llm_outputs": llm_outputs,
                        "human_translation": human_translation,
                        "score": quality_score,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "decision": "reject",
                        "reason": "not better than model outputs",
                    }
                else:
                    is_better = 1.0

        # -----------------------------
        # STEP 4: Compute final score
        # -----------------------------
        # Get best CHRF score (human vs models)
        chrf_score = max(self._chrf(t, human_translation) for t in llm_outputs.values())
        
        # Get best length ratio
        length_score = max(
            self._length_ratio(t, human_translation) 
            for t in llm_outputs.values()
        )
        
        # Adjust is_better based on similarity
        if max_sim > 0.85 and is_better > 0.5:
            # For high similarity, require stronger evidence of improvement
            quality_improvement = self._check_quality_improvement(human_translation, llm_outputs)
            is_better = quality_improvement  # Use the computed improvement score

        metrics = {
            "chrf": chrf_score,
            "length": length_score,
            "llm_judge": float(is_better),
        }

        final_score = sum(metrics[k] * self.weights[k] for k in self.weights)
        
        # Special case: if similarity is high (>0.9) but score is good, still accept
        if max_sim > 0.9 and final_score > 0.8:
            decision = "accept"
        else:
            decision = "accept" if final_score >= self.accept_threshold else "reject"

        result = {
            "source_text": source_text,
            "language-pair": f"{language_pair[0]}-{language_pair[1]}",
            "llm_outputs": llm_outputs,
            "human_translation": human_translation,
            "score": round(float(final_score), 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
        }
        
        if decision == "reject":
            if final_score < self.accept_threshold:
                result["reason"] = f"confidence score below threshold ({final_score:.3f} < {self.accept_threshold})"
            elif max_sim > self.near_threshold:
                result["reason"] = f"high similarity ({max_sim:.3f}) with insufficient improvement"
            else:
                result["reason"] = "not better than model outputs"
        
        return result

import faiss
import numpy as np
import os
import re
import json
from typing import Any
from sentence_transformers import SentenceTransformer

class DeduplicatorAgent:
    """
    FAISS-based deduplicator for human corrections.

    Role:
    - Maintain a vector DB of all accepted human sentences.
    - For each new human_edit, split into sentences, embed, and check
      similarity with existing embeddings.
    - If any sentence is too similar (>= threshold) to a stored one,
      mark the whole correction as duplicate and reject.
    - Otherwise, accept and update FAISS with the new sentence embeddings.
    - Initialize FAISS memory with existing entries in master.jsonl to prevent
      duplicates in fine-tuning queue.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        similarity_threshold: float = 0.85,
        faiss_index_path: str | None = None,
        master_jsonl_path: str | None = None,  # new argument
    ):
        """
        Args:
        embedding_model: A SentenceTransformer instance (re-use validator.embedder).
        similarity_threshold: Cosine similarity threshold for considering
        two sentences duplicates.
        faiss_index_path: Optional path to persist / load FAISS index.
        master_jsonl_path: Optional path to master.jsonl to preload FAISS memory.
        """
        self.embedder = embedding_model
        self.similarity_threshold = similarity_threshold
        self.faiss_index_path = faiss_index_path

        # We will store L2-normalized embeddings to approximate cosine similarity
        self.index: faiss.Index | None = None
        self.dimension: int | None = None
        self.num_vectors: int = 0

        # For simple persistence of raw vectors (optional)
        self._vectors: list[np.ndarray] = []

        # Load existing FAISS index if it exists
        if self.faiss_index_path and os.path.exists(self.faiss_index_path):
            self._load_index()

        # Load human corrections from master.jsonl into FAISS (batch mode)
        if master_jsonl_path and os.path.exists(master_jsonl_path):
            self._load_master_into_faiss(master_jsonl_path)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _build_index(self, dim: int) -> None:
        """Initialize a FAISS index for inner product (cosine on normalized vecs)."""
        self.dimension = dim
        self.index = faiss.IndexFlatIP(dim)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        """L2-normalize vectors along last dimension."""
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def _add_vectors(self, vecs: np.ndarray) -> None:
        """Add vectors to FAISS index (create if needed)."""
        if vecs.size == 0:
            return

        vecs = self._normalize(vecs).astype("float32")

        if self.index is None:
            self._build_index(vecs.shape[1])

        self.index.add(vecs)
        self.num_vectors += vecs.shape[0]
        self._vectors.append(vecs)

    def _search(self, vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Search nearest neighbors for given vectors."""
        if self.index is None or self.num_vectors == 0:
            n = vecs.shape[0]
            return (
                np.zeros((n, 0), dtype="float32"),
                np.zeros((n, 0), dtype="int64"),
            )

        vecs = self._normalize(vecs).astype("float32")
        distances, indices = self.index.search(vecs, k=1)
        return distances, indices

    def _save_index(self) -> None:
        if not self.faiss_index_path or self.index is None:
            return
        faiss.write_index(self.index, self.faiss_index_path)

    def _load_index(self) -> None:
        self.index = faiss.read_index(self.faiss_index_path)
        self.dimension = self.index.d

    def _load_master_into_faiss(self, master_jsonl_path: str) -> None:
        """Preload FAISS memory with human_translation sentences from master.jsonl (batched)."""
        texts: list[str] = []

        with open(master_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                human_text = item.get("human_translation", "").strip()
                if not human_text:
                    continue
                # Segment into sentences to match runtime behaviour
                sentences = self._segment_sentences(human_text)
                texts.extend(sentences)

        if not texts:
            return

        # Batch-encode all sentences once
        emb = self._embed_sentences(texts)
        if emb.size == 0:
            return

        # Add once and save index once
        self._add_vectors(emb)
        self._save_index()
        print(
            f"Deduplicator: Preloaded {emb.shape[0]} sentence embeddings from {master_jsonl_path} "
            f"into FAISS (total={self.num_vectors})"
        )

    # -----------------------------
    # Public API
    # -----------------------------
    def _segment_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        parts = re.split(r"[\.!\?؟]+", text)
        sentences = [s.strip() for s in parts if s.strip()]
        return sentences

    def _embed_sentences(self, sentences: list[str]) -> np.ndarray:
        """Encode sentences into embeddings using the shared embedder."""
        if not sentences:
            return np.zeros((0, self.dimension or 384), dtype="float32")

        emb = self.embedder.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if isinstance(emb, list):
            emb = np.array(emb)
        return emb.astype("float32")

    def is_duplicate(self, human_edit: str) -> bool:
        """Check if any sentence in human_edit is too similar to existing ones."""
        sentences = self._segment_sentences(human_edit)
        if not sentences:
            return False

        sent_emb = self._embed_sentences(sentences)
        if self.index is None or self.num_vectors == 0:
            return False

        distances, _ = self._search(sent_emb)
        if distances.size == 0:
            return False

        max_sim = float(distances.max())
        if max_sim >= self.similarity_threshold:
            print(
                f"Deduplicator: Found duplicate sentence (similarity={max_sim:.3f} ≥ {self.similarity_threshold})"
            )
            return True

        return False

    def update_memory(self, human_edit: str) -> None:
        """Add sentences of accepted human_edit to FAISS index."""
        sentences = self._segment_sentences(human_edit)
        if not sentences:
            return

        sent_emb = self._embed_sentences(sentences)
        self._add_vectors(sent_emb)
        self._save_index()
        print(
            f"Deduplicator: Added {sent_emb.shape[0]} sentence embeddings to FAISS (total={self.num_vectors})"
        )

    def process(self, correction: dict) -> dict:
        """Process a validated correction: reject if duplicate, else accept."""
        human_text = correction.get("human_translation", "").strip()
        if not human_text:
            return correction

        if correction.get("decision") != "accept":
            return correction

        if self.is_duplicate(human_text):
            correction["decision"] = "reject"
            reason = correction.get("reason", "")
            if reason:
                correction["reason"] = reason + "; duplicate human correction"
            else:
                correction["reason"] = "duplicate human correction"
            print("Deduplicator: Correction marked as duplicate -> reject")
            return correction

        self.update_memory(human_text)
        print("Deduplicator: Correction is unique -> keep as accept")
        return correction

class ExecutorAgent:
    """
    Persist validated and deduplicated human translation corrections to disk.
    """

    def __init__(self, output_path: str = "master.jsonl"):
        self.output_path = output_path

        # JSON schema for validation
        self.schema = {
            "type": "object",
            "properties": {
                "source_text": {"type": "string"},
                "language-pair": {"type": "string"},  # stored as "(src, tgt)"
                "llm_outputs": {"type": "object"},
                "human_translation": {"type": "string"},
                "score": {"type": "number"},
                "timestamp": {"type": "string"}
            },
            "required": ["source_text", "language-pair", "llm_outputs",
                         "human_translation", "score", "timestamp"]
        }

        # Create file if it doesn't exist
        if not os.path.exists(self.output_path):
            open(self.output_path, "a", encoding="utf-8").close()

    def persist(self, correction: Dict) -> bool:
        """
        Persist a validated correction to JSONL.
        Returns True if appended, False if schema validation fails.
        """

        # Validate schema
        try:
            validate(instance=correction, schema=self.schema)
            if not correction.get("decision") == "accept":
                print("Executor: Correction not accepted, skipping persistence")
                return False
            else:
                # Remove decision and reason before saving
                correction.pop("decision", None)
                correction.pop("reason", None)
        except ValidationError as e:
            print(f"Schema validation failed: {e.message}")
            return False

        # Append to JSONL
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(correction, ensure_ascii=False) + "\n")

        return True

import threading
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class MonitorAgent:
    """
    Monitor Agent

    Role:
    - Observe accumulation of accepted, unique corrections in master.jsonl.
    - Trigger fine-tuning when thresholds are met:
      * Number of corrections (e.g., 10_000)
      * Time-based threshold since last reset (e.g., 7 days)
    - Manage lifecycle:
      * Lock file and prepare dataset for fine-tuning
      * Provide few-shot examples (last N corrections) for prompting
      * After fine-tuning, reset:
        - Clear FAISS memory (deduplicator)
        - Archive master.jsonl
        - Start fresh for next cycle
    """

    def __init__(
        self,
        path_file_jsonl: str = "master.jsonl",
        deduplicator: Optional[DeduplicatorAgent] = None,
        line_threshold: int = 10_000,
        days_threshold: int = 7,
        few_shot_n: int = 50,
        archive_dir: str = "archive",
        faiss_index_path: str = "dedup_index.faiss",
        state_path: str = "monitor_state.json",
    ):
        """
        Args:
        path_file_jsonl: Path to the master JSONL file.
        deduplicator: The DeduplicatorAgent instance (to clear FAISS).
        line_threshold: Number of corrections to trigger fine-tuning.
        days_threshold: Number of days to trigger fine-tuning.
        few_shot_n: How many latest corrections to use for few-shot.
        archive_dir: Directory where archived master.jsonl copies are stored.
        faiss_index_path: Path of FAISS index file to remove on reset.
        state_path: Path to a small JSON file storing last reset timestamp.
        """
        self.path_file_jsonl = path_file_jsonl
        self.deduplicator = deduplicator
        self.line_threshold = line_threshold
        self.days_threshold = days_threshold
        self.few_shot_n = few_shot_n
        self.archive_dir = archive_dir
        self.faiss_index_path = faiss_index_path
        self.state_path = state_path

        os.makedirs(self.archive_dir, exist_ok=True)

        # Threading lock to avoid race conditions on file operations
        self._lock = threading.Lock()

        # Load or initialize last reset timestamp
        self.last_reset_ts = self._load_last_reset_timestamp()

    # -----------------------------
    # State handling
    # -----------------------------
    def _load_last_reset_timestamp(self) -> datetime:
        """Load last reset timestamp from a small JSON state file."""
        if not os.path.exists(self.state_path):
            # Initialize with now
            ts = datetime.now()
            self._save_last_reset_timestamp(ts)
            return ts

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts_str = data.get("last_reset_ts")
            if not ts_str:
                raise ValueError("Missing last_reset_ts")
            return datetime.fromisoformat(ts_str)
        except Exception:
            # Fallback: use current time
            ts = datetime.now()
            self._save_last_reset_timestamp(ts)
            return ts

    def _save_last_reset_timestamp(self, ts: datetime) -> None:
        """Persist last reset timestamp to JSON."""
        data = {"last_reset_ts": ts.isoformat()}
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # -----------------------------
    # Basic file stats
    # -----------------------------
    def _get_line_count(self) -> int:
        """Return how many lines (corrections) are in master.jsonl."""
        if not os.path.exists(self.path_file_jsonl):
            return 0
        with open(self.path_file_jsonl, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _time_since_last_reset(self) -> timedelta:
        """Return timedelta since last reset."""
        return datetime.now() - self.last_reset_ts

    # -----------------------------
    # Few-shot examples
    # -----------------------------
    def get_last_n_corrections(self, n: Optional[int] = None) -> List[Dict]:
        """
        Read the last N JSON objects from master.jsonl.

        NOTE: Simple implementation: read all lines, slice last N.
        This is fine for moderate sizes; can be optimized later.
        """
        n = n or self.few_shot_n
        if not os.path.exists(self.path_file_jsonl):
            return []

        with open(self.path_file_jsonl, "r", encoding="utf-8") as f:
            lines = f.readlines()

        last_lines = lines[-n:]
        corrections = []
        for line in last_lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                corrections.append(obj)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
        return corrections

    def build_few_shot_prompt(
        self,
        source_text: str,
        model_translation: str,
    ) -> str:
        """
        Build a few-shot prompt using last N corrections + new sentence to translate.

        Format for each example:
        Source: ...
        LLM Translation: ...
        Corrected: ...

        Then append:
        Source: 
        LLM Translation: 
        Corrected:
        """
        examples = self.get_last_n_corrections()
        parts: List[str] = []

        for ex in examples:
            src = ex.get("source_text") or ex.get("source") or ""
            llm_outputs = ex.get("llm_outputs", {})
            # Pick one representative model output (e.g., first)
            llm_translation = ""
            if isinstance(llm_outputs, dict) and llm_outputs:
                # deterministic order: sorted by key
                first_key = sorted(llm_outputs.keys())[0]
                llm_translation = llm_outputs[first_key]
            corrected = ex.get("human_translation") or ""

            parts.append(
                f"Source: {src}\n"
                f"LLM Translation: {llm_translation}\n"
                f"Corrected: {corrected}\n"
            )

        # Append the new item to translate
        parts.append(
            f"Source: {source_text}\n"
            f"LLM Translation: {model_translation}\n"
            f"Corrected:"
        )

        return "\n".join(parts)

    # -----------------------------
    # Trigger logic
    # -----------------------------
    def should_trigger_finetune(self) -> bool:
        """
        Decide whether fine-tuning should be triggered based on:
        - line_threshold (number of corrections)
        - days_threshold (time since last reset)
        """
        line_count = self._get_line_count()
        elapsed = self._time_since_last_reset()

        print(
            f"Monitor: master.jsonl has {line_count} lines, "
            f"time since last reset: {elapsed.days} days"
        )

        if line_count >= self.line_threshold:
            print(
                f"Monitor: Line threshold reached "
                f"({line_count} >= {self.line_threshold})"
            )
            return True

        if elapsed >= timedelta(days=self.days_threshold):
            print(
                f"Monitor: Time threshold reached "
                f"({elapsed.days} >= {self.days_threshold} days)"
            )
            return True

        return False

    # -----------------------------
       # Fine-tuning preparation / reset
    # -----------------------------
    def prepare_finetune_dataset(self, output_path: str = "finetune_dataset.jsonl") -> str:
        """
        Lock master.jsonl and copy it to a temporary location for fine-tuning.

        Returns:
        Path of the dataset file to use for fine-tuning.
        """
        with self._lock:
            if not os.path.exists(self.path_file_jsonl):
                print("Monitor: No master.jsonl found, nothing to prepare.")
                return ""

            shutil.copy2(self.path_file_jsonl, output_path)
            print(f"Monitor: Prepared fine-tuning dataset at {output_path}")
            return output_path

    def _reset_faiss_memory(self) -> None:
        """
        Clear FAISS memory used by the DeduplicatorAgent.
        Simple strategy: reinitialize the index and delete the index file.
        """
        if self.deduplicator is None:
            print("Monitor: No deduplicator provided, skipping FAISS reset.")
            return

        # Reset in-memory index
        self.deduplicator.index = None
        self.deduplicator.dimension = None
        self.deduplicator.num_vectors = 0
        self.deduplicator._vectors = []

        # Remove on-disk index file, if present
        if self.faiss_index_path and os.path.exists(self.faiss_index_path):
            os.remove(self.faiss_index_path)
            print(f"Monitor: Removed FAISS index file {self.faiss_index_path}")

        print("Monitor: FAISS memory cleared.")

    def archive_and_reset(self) -> str:
        """
        Archive current master.jsonl, clear FAISS memory, and start fresh.

        Returns:
        Path of the archived file.
        """
        with self._lock:
            if not os.path.exists(self.path_file_jsonl):
                print("Monitor: No master.jsonl to archive.")
                return ""

            # Archive master.jsonl with timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(
                self.archive_dir, f"master_{ts}.jsonl"
            )
            shutil.move(self.path_file_jsonl, archive_path)
            print(f"Monitor: Archived master.jsonl to {archive_path}")

            # Reset FAISS / deduplicator (current cycle)
            self._reset_faiss_memory()

            # Update last reset timestamp
            self.last_reset_ts = datetime.now()
            self._save_last_reset_timestamp(self.last_reset_ts)

            # Create fresh empty master.jsonl
            open(self.path_file_jsonl, "w", encoding="utf-8").close()
            print("Monitor: Created new empty master.jsonl")

            # IMPORTANT: re-instantiate deduplicator to start next cycle cleanly
            if self.deduplicator is not None:
                # reuse same embedder and config
                embedder = self.deduplicator.embedder
                sim_thr = self.deduplicator.similarity_threshold
                self.deduplicator = DeduplicatorAgent(
                    embedding_model=embedder,
                    similarity_threshold=sim_thr,
                    faiss_index_path=self.faiss_index_path,
                    master_jsonl_path=self.path_file_jsonl,
                )
                print("Monitor: Re-instantiated DeduplicatorAgent for new cycle.")

            return archive_path

    # -----------------------------
    # High-level entrypoint
    # -----------------------------
    def check_and_handle_cycle(self) -> Optional[str]:
        """
        High-level call to be used periodically (or after each batch):
        - Check thresholds
        - If triggered:
          * prepare dataset
          * (you run fine-tuning externally on that dataset)
          * archive and reset after fine-tuning

        Returns:
        Path to the dataset file if fine-tuning should start,
        None otherwise.
        """
        if not self.should_trigger_finetune():
            return None

        # Step 1: prepare dataset
        dataset_path = self.prepare_finetune_dataset("finetune_dataset.jsonl")

        if not dataset_path:
            return None

        print(
            "Monitor: Fine-tuning should be started externally using "
            f"dataset at {dataset_path}"
        )
        # In your workflow, you would:
        # - run fine-tuning using dataset_path
        # - then call archive_and_reset() once fine-tuning finishes

        return dataset_path

# -----------------------------
# Initialize agents
# -----------------------------
# -----------------------------
# Initialize agents
# -----------------------------
path_file_jsonl = "master.jsonl"
preprocessor = PreprocessorAgent()
validator = ValidatorAgent()

# Deduplicator Agent: reuse validator's embedding model
deduplicator = DeduplicatorAgent(
    embedding_model=validator.embedder,
    similarity_threshold=0.85,
    faiss_index_path="dedup_index.faiss",
    master_jsonl_path="master.jsonl",
)

executor = ExecutorAgent(path_file_jsonl)

# Monitor Agent
monitor = MonitorAgent(
    path_file_jsonl=path_file_jsonl,
    deduplicator=deduplicator,
    line_threshold=2,  # you can tune this
    days_threshold=7,       # you can tune this
    few_shot_n=50,
    archive_dir="archive",
    faiss_index_path="dedup_index.faiss",
    state_path="monitor_state.json",
)



# -----------------------------
# Define test cases
# -----------------------------
test_cases = [
    # 1. Minor cosmetic change (should reject)
    {
        "source": "Good morning, everyone.",
        "llm_outputs": {
            "model1": "صباح الخير جميعا.",
            "model2": "أهلا صباح الخير للجميع.",
            "model3": "صباح الخير يا الجميع."
        },
        "human_edit": "صباح الخير جميعاً.",
        "expected": "reject"
    },
    # 2. Adds something not in the source
    {
        "source": "The quick brown fox jumps over the lazy dog.",
        "llm_outputs": {
            "model1": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
            "model2": "الثعلب البني يقفز على الكلب الكسول بسرعة.",
            "model3": "الثعلب البني السريع يطير فوق الكلب الكسول."
        },
        "human_edit": "الثعلب البني السريع يقفز على الكلب الكسول بطريقة طبيعية.",
        "expected": "reject"
    },
    # 3. Wrong translation (should reject)
    {
        "source": "I like to eat apples.",
        "llm_outputs": {
            "model1": "أنا أحب أكل التفاح.",
            "model2": "أحب أن أتناول التفاح.",
            "model3": "أفضل تناول التفاح."
        },
        "human_edit": "أنا أحب أكل الموز.",
        "expected": "reject"
    },
    # 4. New terminology (should accept if judged better)
    {
        "source": "The server crashed due to memory overflow.",
        "llm_outputs": {
            "model1": "تعطل الخادم بسبب زيادة في الذاكرة.",
            "model2": "تعطل السيرفر بسبب تجاوز الذاكرة.",
            "model3": "توقف الخادم عن العمل نتيجة امتلاء الذاكرة."
        },
        "human_edit": "انهار الخادم بسبب فيضان الذاكرة.",
        "expected": "accept"
    },
    # 5. Minor rewording across outputs (should reject)
    {
        "source": "Please submit your report by Friday.",
        "llm_outputs": {
            "model1": "يرجى تقديم تقريرك بحلول يوم الجمعة.",
            "model2": "من فضلك أرسل تقريرك قبل الجمعة.",
            "model3": "تأكد من تقديم تقريرك يوم الجمعة."
        },
        "human_edit": "يرجى تقديم تقريرك قبل يوم الجمعة.",
        "expected": "reject"
    },
    # 6. Identical to one model output (should reject)
    {
        "source": "Hello world, this is a test.",
        "llm_outputs": {
            "model1": "مرحبا بالعالم، هذا اختبار.",
            "model2": "أهلا بالعالم، هذا اختبار.",
            "model3": "سلام عالم، هذا تجربة."
        },
        "human_edit": "مرحبا بالعالم، هذا اختبار.",
        "expected": "reject"
    },
    # 7. Duplicate of Test 2 (should be rejected by deduplicator if 2 was accepted)
    {
        "source": "The quick brown fox jumps over the lazy dog.",
        "llm_outputs": {
            "model1": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
            "model2": "الثعلب البني يقفز على الكلب الكسول بسرعة.",
            "model3": "الثعلب البني السريع يطير فوق الكلب الكسول."
        },
        "human_edit": "الثعلب البني السريع يقفز على الكلب الكسول بطريقة طبيعية.",
        # Expected from deduplicator point of view: reject as duplicate
        "expected": "reject"
    },
]




# -----------------------------
# Run all tests
# -----------------------------
for i, test in enumerate(test_cases, 1):
    print(f"\n================ Test {i} =================")
    processed = preprocessor.process(test["source"], test["llm_outputs"], test["human_edit"])
    if processed is None:
        print("Preprocessing failed.")
        continue
    
    correction = validator.validate(
        source_text=processed["source"],
        language_pair=("en", "ar"),
        llm_outputs=processed["llm_outputs"],
        human_translation=processed["human_edit"]
    )

    # Pass through DeduplicatorAgent AFTER validation
    correction = deduplicator.process(correction)

    print("Validation output:")
    print(correction)
    print("Expected decision:", test["expected"])
    print("Test PASSED:", correction["decision"] == test["expected"])

    # Persist to JSONL
    executor.persist(correction)




# -----------------------------
# Check JSONL content
# -----------------------------
with open(path_file_jsonl, "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(f"\nJSONL file now has {len(lines)} entries")
    if lines:
        print("Last entry in JSONL:")
        print(lines[-1])


# Optionally, check if fine-tuning should be triggered
dataset_path = monitor.check_and_handle_cycle()
if dataset_path:
    print(f"\nMonitor suggests starting fine-tuning with dataset: {dataset_path}")
    # to finetune:
    #monitor.archive_and_reset()

class OrchestratorAgent:
    """
    Orchestrator Agent

    Role:
    - Initialize and hold all pipeline agents (Preprocessor, Validator,
      Deduplicator, Executor, Monitor).
    - Expose a simple process_batch(frontend_payload) method that takes
      a list of dicts of the form:
        [
            {
                "source": "...",
                "llm_outputs": {...},
                "human_edit": "...",
                "expected": "accept" / "reject"  # optional, for testing
            },
            ...
        ]
    - For each item:
      1) Preprocess (normalize/structure)
      2) Validate (quality/meaning)
      3) Deduplicate (FAISS)
      4) Persist accepted corrections to master.jsonl
      5) Optionally check Monitor to see if fine-tuning should start
    """

    def __init__(
        self,
        path_file_jsonl: str = "master.jsonl",
        faiss_index_path: str = "dedup_index.faiss",
        archive_dir: str = "archive",
        monitor_state_path: str = "monitor_state.json",
        line_threshold: int = 10_000,
        days_threshold: int = 7,
        few_shot_n: int = 50,
    ):
        # Initialize base agents (same logic and names as before)
        self.path_file_jsonl = path_file_jsonl

        self.preprocessor = PreprocessorAgent()
        self.validator = ValidatorAgent()

        # Deduplicator reuses validator's embedding model
        self.deduplicator = DeduplicatorAgent(
            embedding_model=self.validator.embedder,
            similarity_threshold=0.85,
            faiss_index_path=faiss_index_path,
            master_jsonl_path=self.path_file_jsonl,
        )

        self.executor = ExecutorAgent(self.path_file_jsonl)

        self.monitor = MonitorAgent(
            path_file_jsonl=self.path_file_jsonl,
            deduplicator=self.deduplicator,
            line_threshold=line_threshold,
            days_threshold=days_threshold,
            few_shot_n=few_shot_n,
            archive_dir=archive_dir,
            faiss_index_path=faiss_index_path,
            state_path=monitor_state_path,
        )

    def process_batch(
        self,
        frontend_payload: list[dict],
        language_pair: tuple[str, str] = ("en", "ar"),
        check_monitor: bool = True,
    ) -> dict:
        """
        Run the full pipeline on a batch of changes coming from the frontend.

        Args:
        - frontend_payload: list of dicts with keys:
            "source": str
            "llm_outputs": dict[str, str]
            "human_edit": str
            (optional) "expected": "accept"/"reject" (used only for testing)
        - language_pair: translation direction, default ("en", "ar").
        - check_monitor: whether to call monitor.check_and_handle_cycle() after batch.

        Returns:
        - summary dict with:
            "results": list of per-item outputs (corrections + optional test info)
            "dataset_path": path to finetune dataset if monitor fired, else None
        """
        results: list[dict] = []
        any_accepted = False  # track if at least one correction was accepted

        for i, item in enumerate(frontend_payload, 1):
            source = item.get("source", "")
            llm_outputs = item.get("llm_outputs", {})
            human_edit = item.get("human_edit", "")
            expected = item.get("expected")  # may be None

            print(f"\n================ Orchestrator item {i} =================")
            print(f"Source: {source}")
            print(f"Human edit: {human_edit}")

            try:
                # 1. Preprocess
                processed = self.preprocessor.process(source, llm_outputs, human_edit)
                if processed is None:
                    print("Preprocessor: Processing failed, skipping item.")
                    results.append(
                        {
                            "index": i,
                            "status": "preprocess_failed",
                            "reason": "Preprocessor returned None",
                            "expected": expected,
                        }
                    )
                    continue

                # 2. Validate
                correction = self.validator.validate(
                    source_text=processed["source"],
                    language_pair=language_pair,
                    llm_outputs=processed["llm_outputs"],
                    human_translation=processed["human_edit"],
                )

                # 3. Deduplicate AFTER validation
                correction = self.deduplicator.process(correction)

                print("Orchestrator: Validation + deduplication output:")
                print(correction)

                # 4. Persist to JSONL
                self.executor.persist(correction)

                # Track if accepted
                if correction.get("decision") == "accept":
                    any_accepted = True

                # 5. For testing: compare with expected if provided
                test_passed = None
                if expected is not None and "decision" in correction:
                    test_passed = (correction["decision"] == expected)
                    print("Expected decision:", expected)
                    print("Test PASSED:", test_passed)

                results.append(
                    {
                        "index": i,
                        "correction": correction,
                        "expected": expected,
                        "test_passed": test_passed,
                    }
                )

            except Exception as e:
                print(f"Orchestrator: Error processing item {i}: {e}")
                results.append(
                    {
                        "index": i,
                        "status": "error",
                        "reason": str(e),
                        "expected": expected,
                    }
                )

        dataset_path = None
        if check_monitor:
            if any_accepted:
                dataset_path = self.monitor.check_and_handle_cycle()
                if dataset_path:
                    print(
                        f"\nMonitor suggests starting fine-tuning with dataset: {dataset_path}"
                    )
            else:
                # No correction was accepted in the whole batch
                dataset_path = "no correction was accepted"

        return {"results": results, "dataset_path": dataset_path}

# -----------------------------
# Example frontend payload
# -----------------------------
frontend_payload = [
    {
        "source": "Good morning, everyone.",
        "llm_outputs": {
            "model1": "صباح الخير جميعا.",
            "model2": "أهلا صباح الخير للجميع.",
            "model3": "صباح الخير يا الجميع."
        },
        "human_edit": "صباح الخير جميعاً.",
        "expected": "accept",
    },
    {
        "source": "The quick brown fox jumps over the lazy dog.",
        "llm_outputs": {
            "model1": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
            "model2": "الثعلب البني يقفز على الكلب الكسول بسرعة.",
            "model3": "الثعلب البني السريع يطير فوق الكلب الكسول."
        },
        "human_edit": "الثعلب البني السريع يقفز على الكلب الكسول بطريقة طبيعية.",
        "expected": "reject",
    },
    {
        "source": "The meeting was postponed to next Monday.",
        "llm_outputs": {
            "model1": "تم تأجيل الاجتماع إلى الاثنين القادم.",
            "model2": "تأجل الاجتماع حتى الاثنين المقبل.",
            "model3": "أُجّل الاجتماع للاثنين القادم."
        },
        "human_edit": "تم تأجيل الاجتماع إلى يوم الاثنين المقبل.",
        "expected": "accept",
    },
]

# -----------------------------
# Initialize orchestrator
# -----------------------------
orchestrator = OrchestratorAgent(
    path_file_jsonl="master.jsonl",
    faiss_index_path="dedup_index.faiss",
    archive_dir="archive",
    monitor_state_path="monitor_state.json",
    line_threshold=4,  # same as before
    days_threshold=7,
    few_shot_n=50,
)

# -----------------------------
# Run orchestrator on payload
# -----------------------------
orchestrator_output = orchestrator.process_batch(
    frontend_payload=frontend_payload,
    language_pair=("en", "ar"),
    check_monitor=True,
)

print("\n===== Orchestrator summary =====")
for item_result in orchestrator_output["results"]:
    idx = item_result["index"]
    expected = item_result.get("expected")

    correction = item_result.get("correction")
    if correction is None:
        status = item_result.get("status", "no_correction")
        print(f"Item {idx}: status={status}, expected={expected}, test_passed={item_result.get('test_passed')}")
        continue

    decision = correction.get("decision")
    print(f"Item {idx}: decision={decision}, expected={expected}, test_passed={item_result['test_passed']}")

print("\nDataset path:", orchestrator_output["dataset_path"])
