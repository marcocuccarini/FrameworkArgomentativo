from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class T5_Model:

    def __init__(self, tokenizer, model_name, token):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token)

    def generate_title(self, text, repetition_penalty=15.0, diversity_penalty=1.0):
        inputs = self.tokenizer(
            f'paraphraser: {text}',
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=64
        )
        
        outputs = self.model.generate(
            inputs.input_ids,
            num_beams=4,
            num_beam_groups=4,  # Usa Group Beam Search
            num_return_sequences = 1,
            repetition_penalty= repetition_penalty,
            diversity_penalty= diversity_penalty,
            no_repeat_ngram_size=2,
            max_length=64,
            temperature=0.8             # Controlla la casualità

        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)    