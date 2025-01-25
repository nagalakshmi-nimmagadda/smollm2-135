import gradio as gr
import torch
from transformers import AutoTokenizer
from model import SmolLM2ForCausalLM
from config import SmolLM2Config

# Initialize model and tokenizer
def load_model():
    try:
        config = SmolLM2Config()
        model = SmolLM2ForCausalLM(config)
        
        # Load the trained weights
        checkpoint = torch.load('checkpoints/step_5050.pt', map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Fix the state dict keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.model.'):
                # Remove one 'model.' prefix
                new_key = key.replace('model.model.', 'model.')
                new_state_dict[new_key] = value
            elif key.startswith('model.lm_head.'):
                # Fix lm_head key
                new_key = key.replace('model.lm_head.', 'lm_head.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load the fixed state dict
        model.load_state_dict(new_state_dict)
        model.eval()
        
        # Initialize tokenizer with padding token
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/cosmo2-tokenizer",
            revision=None,
            use_fast=True
        )
        # Set padding token to be the same as EOS token
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Text generation function
def generate_text(prompt, max_tokens=50, temperature=0.7):
    try:
        if not prompt:
            return "Please enter a prompt."
            
        model, tokenizer = load_model()
        
        with torch.no_grad():
            # Use padding=True only if needed
            input_ids = tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Add max length to prevent too long inputs
            )["input_ids"]
            
            output_sequence = input_ids.clone()
            
            for _ in range(int(max_tokens)):
                outputs, _ = model(input_ids=output_sequence)
                next_token_logits = outputs[:, -1, :] / float(temperature)
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                output_sequence = torch.cat([output_sequence, next_token], dim=-1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
            generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
            return generated_text
            
    except Exception as e:
        return f"Error generating text: {str(e)}"

# Create Gradio interface
def create_interface():
    # Example prompts as a list of dictionaries
    examples = [
        ["My bounty is as boundless as the sea, my love as deep; the more I give to thee, the more I have, for both are infinite.", 50, 0.7],
        ["All the world’s a stage, and all the men and women merely players.", 50, 0.7],
        ["The course of true love never did run smooth.", 50, 0.7],
        ["We are such stuff as dreams are made on, and our little life is rounded with a sleep.", 50, 0.7]
    ]
    
    interface = gr.Interface(
        fn=generate_text,
        inputs=[
            gr.Textbox(
                label="Enter your prompt",
                placeholder="Type your prompt here...",
                lines=2
            ),
            gr.Slider(
                minimum=10,
                maximum=200,
                value=50,
                step=10,
                label="Max Tokens",
                info="Controls the length of generated text"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Controls randomness (higher = more random)"
            )
        ],
        outputs=gr.Textbox(
            label="Generated Text",
            lines=5
        ),
        title="SmolLM2-135M Shakespeare Text Generator",
        description="""This is a 135M parameter language model trained on Shakespeare's text.
        Enter a prompt and adjust the generation parameters to see the model's output.
        Click on any example below to try it out!""",
        examples=examples,
        cache_examples=False,
        allow_flagging="never"
    )
    
    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=False,
        debug=True,
        show_error=True
    ) 