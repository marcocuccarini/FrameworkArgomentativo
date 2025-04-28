from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class T5_Model:

    def __init__(self, tokenizer, model_name, token):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token)

    def generate_title(self, text, repetition_penalty=2.5, diversity_penalty=7.0):
        inputs = self.tokenizer(
            f'paraphraser: {text}',
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=64
        )

        outputs = self.model.generate(
            inputs.input_ids,
            num_beams=20,  # Standard beam search (no group beam search)
            num_beam_groups=2,  # Set to 1 to disable Group Beam Search
            num_return_sequences=10,
            repetition_penalty=repetition_penalty,
            diversity_penalty=diversity_penalty,
            no_repeat_ngram_size=2,
            max_length=64,
            do_sample=False,  # Enable sampling for diversity
            temperature=1.2,
            top_k=50,
            top_p=0.9
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

