import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import random
from openai import OpenAI  # Ensure you have the correct OpenAI client installed
import json
import logging
import time
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
from tokenizers import BertWordPieceTokenizer
import gradio as gr
from torchtext.vocab import GloVe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# 1. Enhanced Fractal Neuron Model Components
# -----------------------------

class BrocasModule(nn.Module):
    def __init__(self, gpt_model_name='gpt2'):
        super(BrocasModule, self).__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        self.gpt.eval()  # Set to evaluation mode

    def forward(self, input_ids, attention_mask=None, max_new_tokens=50, temperature=0.7):
        output_sequences = self.gpt.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,  # Specify number of new tokens to generate
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=self.gpt.config.eos_token_id
        )
        return output_sequences

class WernickesModule(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_intents=10, num_entities=10):
        super(WernickesModule, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.entity_extractor = nn.Linear(self.bert.config.hidden_size, num_entities)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        intent_logits = self.intent_classifier(cls_output)
        entity_logits = self.entity_extractor(outputs.last_hidden_state)
        return intent_logits, entity_logits

### 1.1 Synapse
class Synapse(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Synapse, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.plasticity = 0.1  # Learning rate for synapse updates

    def forward(self, x):
        # x: (batch_size, input_dim)
        # self.weight: (input_dim, output_dim)
        # Output: (batch_size, output_dim)
        return torch.matmul(x, self.weight)

    def update_weight(self, gradient):
        with torch.no_grad():
            self.weight += self.plasticity * gradient

### 1.2 Soma
class Soma(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Soma, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activation_threshold = 0.5
        self.activity = 0.0

    def forward(self, x):
        # x: (batch_size, input_dim)
        # self.weights: (input_dim, output_dim)
        # self.bias: (output_dim,)
        output = torch.matmul(x, self.weights) + self.bias
        self.activity = torch.mean(output).item()
        return output if self.activity > self.activation_threshold else torch.zeros_like(output)

### 1.3 Axon
class Axon(nn.Module):
    def __init__(self, output_dim):
        super(Axon, self).__init__()
        self.output_dim = output_dim

    def forward(self, x):
        return x  # Placeholder for more complex processing if needed

### 1.4 EmotionalModule
class EmotionalModule(nn.Module):
    def __init__(self, num_emotions=5, input_dim=300):
        super(EmotionalModule, self).__init__()
        self.emotion_embeddings = nn.Parameter(torch.randn(num_emotions, input_dim))
        self.current_emotion = nn.Parameter(torch.zeros(num_emotions), requires_grad=False)

    def update_emotion(self, input_signal):
        # Update current emotion based on input
        with torch.no_grad():
            self.current_emotion += 0.1 * (torch.tanh(input_signal) - self.current_emotion)

    def modulate(self, x):
        # Apply emotional modulation to input
        emotion_vector = F.softmax(self.current_emotion, dim=0)  # (num_emotions,)
        emotional_context = torch.matmul(emotion_vector, self.emotion_embeddings)  # (input_dim,)
        return x + emotional_context  # Broadcasting to (batch_size, input_dim)

### 1.5 EnhancedEmotionalModule
class EnhancedEmotionalModule(EmotionalModule):
    def __init__(self, num_emotions=5, input_dim=300):
        super().__init__(num_emotions, input_dim)
        # Additional parameters or layers can be added here for more nuanced effects

    def update_emotion(self, input_signal):
        # More complex emotion update logic can be implemented here
        super().update_emotion(input_signal)

### 1.6 CuriosityModule
class CuriosityModule(nn.Module):
    def __init__(self, input_dim):
        super(CuriosityModule, self).__init__()
        self.novelty_threshold = 0.5
        self.recent_inputs = []

    def evaluate_novelty(self, x):
        if not self.recent_inputs:
            return True
        x_norm = torch.norm(x, dim=1, keepdim=True)  # (batch_size, 1)
        novelty = 1.0
        for recent in self.recent_inputs:
            similarity = torch.matmul(x, recent.t()) / (x_norm * torch.norm(recent, dim=1, keepdim=True))
            similarity = similarity.clamp(min=1e-5)  # Prevent division by zero
            novelty = min(novelty, (1 - similarity.mean()).item())
        return novelty > self.novelty_threshold

    def update(self, x):
        self.recent_inputs.append(x.detach().clone())
        if len(self.recent_inputs) > 100:
            self.recent_inputs.pop(0)

### 1.7 AttentionMechanism
class AttentionMechanism(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMechanism, self).__init__()
        self.attention = nn.Linear(input_dim, input_dim)

    def forward(self, inputs):
        # inputs: (batch_size, input_dim)
        attention_weights = F.softmax(self.attention(inputs), dim=1)  # (batch_size, input_dim)
        return inputs * attention_weights  # (batch_size, input_dim)

### 1.8 FractalNeuron
class FractalNeuron(nn.Module):
    def __init__(self, input_dim, output_dim, max_children=3, depth=0, max_depth=5):
        super(FractalNeuron, self).__init__()
        self.dendrites = nn.ModuleList([Synapse(input_dim, output_dim) for _ in range(5)])
        self.soma = Soma(output_dim, output_dim)
        self.axon = Axon(output_dim)
        self.attention = AttentionMechanism(output_dim)
        self.emotional_module = EnhancedEmotionalModule(num_emotions=5, input_dim=output_dim)
        self.child_neurons = nn.ModuleList()
        self.max_children = max_children
        self.depth = depth
        self.max_depth = max_depth

    def forward(self, x):
        # x: (batch_size, output_dim)
        dendritic_input = sum(d(x) for d in self.dendrites)  # (batch_size, output_dim)
        soma_output = self.soma(dendritic_input)  # (batch_size, output_dim)
        soma_output = self.emotional_module.modulate(soma_output)  # (batch_size, output_dim)
        attention_output = self.attention(soma_output)  # (batch_size, output_dim)

        axon_output = self.axon(attention_output)  # (batch_size, output_dim)

        if self.child_neurons and isinstance(self.child_neurons, nn.ModuleList):
            child_outputs = [child(axon_output) for child in self.child_neurons]  # List of (batch_size, output_dim)
            return sum(child_outputs) / len(child_outputs)  # (batch_size, output_dim)
        return axon_output  # (batch_size, output_dim)

    def grow(self):
        if len(self.child_neurons) < self.max_children and self.depth < self.max_depth:
            child = FractalNeuron(self.soma.weights.size(1), self.soma.weights.size(1),
                                  depth=self.depth + 1, max_depth=self.max_depth)
            self.child_neurons.append(child)
            self.add_module(f'child_{len(self.child_neurons)}', child)

    def prune(self):
        if self.child_neurons:
            # Simple pruning: remove the child with the lowest activity
            activities = [child.soma.activity for child in self.child_neurons]
            if activities:
                least_active = activities.index(min(activities))
                del self.child_neurons[least_active]
                delattr(self, f'child_{least_active + 1}')

### 1.9 FractalBrain
class FractalBrain(nn.Module):
    def __init__(self, input_dim, output_dim, max_depth=5):
        super(FractalBrain, self).__init__()
        self.root = FractalNeuron(input_dim, output_dim, max_depth=max_depth)
        self.emotional_module = EnhancedEmotionalModule(num_emotions=5, input_dim=output_dim)
        self.curiosity_module = CuriosityModule(output_dim)
        self.attention_mechanism = AttentionMechanism(output_dim)
        self.long_term_memory = LongTermMemory(memory_size=1000, key_size=output_dim, value_size=output_dim)
        self.max_depth = max_depth

    def forward(self, x):
        # x: (batch_size, latent_dim)
        emotional_modulation = self.emotional_module.modulate(x)  # (batch_size, latent_dim)
        output = self.root(emotional_modulation)  # (batch_size, output_dim)
        if self.curiosity_module.evaluate_novelty(x):
            self.grow()
        return output

    def grow(self):
        self.root.grow()

    def prune(self):
        self.root.prune()

    def learn(self, x, y, learning_rate=0.01):
        output = self.forward(x)
        loss = (output - y).pow(2).mean()
        loss.backward()

        # Update weights
        with torch.no_grad():
            for param in self.parameters():
                param -= learning_rate * param.grad
            # Zero gradients after update
            self.zero_grad()

        self.emotional_module.update_emotion(torch.mean(output - y))
        self.curiosity_module.update(x)

    def interactive_learning(self, input_text, tokenizer, embedding_layer, index_to_word):
        # Convert input_text to numerical representation (e.g., using embeddings)
        x = self.text_to_embedding(input_text, tokenizer, embedding_layer)
        output = self.forward(x)
        # Convert output to text
        response = self.embedding_to_text(output, index_to_word)
        return response

    def text_to_embedding(self, text, tokenizer, embedding_layer):
        tokens = tokenizer(text)  # (batch_size, seq_length)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        embedded = embedding_layer(tokens)  # (batch_size, seq_length, embed_dim)
        return embedded.mean(dim=1)  # (batch_size, embed_dim)

    def embedding_to_text(self, embedding, index_to_word):
        # Convert an embedding vector to the closest word in the embedding space
        with torch.no_grad():
            # Normalize the embedding
            embedding_norm = F.normalize(embedding, p=2, dim=1)  # Shape: (1, embed_dim)
            
            # Normalize all word embeddings
            word_embeddings_norm = F.normalize(self.word_embeddings.weight, p=2, dim=1)  # Shape: (vocab_size, embed_dim)
            
            # Compute cosine similarity
            similarities = torch.matmul(embedding_norm, word_embeddings_norm.t())  # Shape: (1, vocab_size)
            
            # Get the index of the highest similarity
            best_match_idx = torch.argmax(similarities, dim=1).item()
            
            # Retrieve the corresponding word
            word = index_to_word.get(best_match_idx, "[UNKNOWN]")
            
            return word

    def save_state(self, filename):
        state = {
            'word_embeddings_state': self.word_embeddings.state_dict(),
            'vae_state': self.vae.state_dict(),
            'brain_state': self.brain.state_dict(),
            'wernickes_state': self.wernickes.state_dict(),
            'brocas_state': self.brocas.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'word_to_index': self.word_to_index,
            'index_to_word': self.index_to_word,
            'next_index': self.next_index,
            'long_term_memory': self.long_term_memory.memory.data
        }
        torch.save(state, filename)
        logger.info(f"Model state saved to {filename}")

    def load_state(self, filename):
        state = torch.load(filename)
        self.word_to_index = state['word_to_index']
        self.index_to_word = state['index_to_word']
        self.next_index = state['next_index']
        self.word_embeddings.load_state_dict(state['word_embeddings_state'])
        self.vae.load_state_dict(state['vae_state'])
        self.brain.load_state_dict(state['brain_state'])
        self.wernickes.load_state_dict(state['wernickes_state'])
        self.brocas.load_state_dict(state['brocas_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
        self.long_term_memory.memory.data = state.get(
            'long_term_memory', 
            torch.randn(1000, self.brain.root.soma.weights.size(1) + self.brain.root.soma.weights.size(1))
        )
        logger.info(f"Model state loaded from {filename}")

    def train_on_qa_pairs(self, qa_pairs, epochs=10):
        if not isinstance(qa_pairs, list) or len(qa_pairs) == 0:
            raise ValueError("qa_pairs must be a non-empty list")

        logger.info(f"Training on {len(qa_pairs)} Q&A pairs for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0
            errors = 0
            random.shuffle(qa_pairs)
            for i, (question, answer) in enumerate(qa_pairs):
                self.optimizer.zero_grad()

                try:
                    # Tokenize question and answer
                    q_input_ids, q_attention_mask = self.tokenize_input(question)
                    a_input_ids, a_attention_mask = self.tokenize_input(answer)

                    # Wernicke's Comprehension for Question
                    q_intent_logits, q_entity_logits = self.wernickes(q_input_ids, q_attention_mask)
                    q_intent = torch.argmax(q_intent_logits, dim=1)
                    q_entities = torch.argmax(q_entity_logits, dim=2)

                    # Convert question to embeddings
                    q_embedded = self.word_embeddings(q_input_ids)  # (1, seq_length, embed_dim)
                    q_embedded = q_embedded.mean(dim=1)  # (1, embed_dim)

                    # VAE Encoding and Decoding for Question
                    q_recon, q_mu, q_logvar, z_q = self.vae(q_embedded)
                    q_latent_output = self.brain.forward(z_q)  # (1, output_dim)
                    q_decoded_output = self.vae.decoder(q_latent_output)  # Decode back to embedding space

                    # VAE Loss for Question
                    vae_loss_q = self.vae.vae_loss(q_recon, q_embedded, q_mu, q_logvar)

                    # Convert answer to embeddings
                    a_embedded = self.word_embeddings(a_input_ids)  # (1, seq_length, embed_dim)
                    a_embedded = a_embedded.mean(dim=1)  # (1, embed_dim)

                    # VAE Encoding and Decoding for Answer
                    a_recon, a_mu, a_logvar, a_latent = self.vae(a_embedded)
                    a_latent_output = self.brain.forward(a_latent)  # (1, output_dim)
                    a_decoded_output = self.vae.decoder(a_latent_output)  # Decode back to embedding space

                    # VAE Loss for Answer
                    vae_loss_a = self.vae.vae_loss(a_recon, a_embedded, a_mu, a_logvar)

                    # Calculate Loss
                    loss = self.criterion(q_decoded_output, a_decoded_output) + vae_loss_q + vae_loss_a
                    loss += FractalRegularization(lambda_param=0.01)(self.brain)
                    loss.backward()

                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.word_embeddings.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.wernickes.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.brocas.parameters(), max_norm=1.0)

                    # Update Parameters
                    self.optimizer.step()

                    total_loss += loss.item()

                    # Dynamic Growth and Pruning
                    self.brain.grow()
                    self.brain.prune()
                    self.brain.emotional_module.update_emotion(torch.mean(loss).item())
                    self.brain.curiosity_module.update(z_q)
                    self.long_term_memory.store(z_q, a_latent)
                    self.continuous_learner.update((z_q, a_latent))

                    if i % 10 == 0:
                        logger.info(f"Epoch {epoch+1}, Pair {i+1}/{len(qa_pairs)}, Loss: {loss.item():.4f}")

                except Exception as e:
                    logger.error(f"Error processing pair: {question} | {answer}")
                    logger.error(f"Error details: {str(e)}")
                    errors += 1
                    continue

            avg_loss = total_loss / (len(qa_pairs) - errors) if len(qa_pairs) > errors else 0
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Errors: {errors}")

            # Adjust learning rate based on performance
            novelty = self.brain.curiosity_module.evaluate_novelty(z_q)
            self.adaptive_lr.adjust(performance_metric=avg_loss, novelty_score=novelty)

            self.scheduler.step()
            self.save_state(f"model_state_epoch_{epoch+1}.pth")

            yield epoch + 1, avg_loss, errors

    def fractal_thinking(self, input_vector):
        # This method is now integrated into the EnhancedFractalBrain's forward method
        pass

    def talk_with_lm_studio(self, initial_message, conversation_duration=60, delay=2):
        message = initial_message
        start_time = time.time()
        conversation_log = []

        while time.time() - start_time < conversation_duration:
            ai_response = self.chat(message)
            logger.info(f"DynamicAI:\n{ai_response}")
            conversation_log.append(f"DynamicAI:\n{ai_response}")
            yield "\n\n".join(conversation_log)

            ai_message = ai_response.split("Response: ")[-1].strip()

            if not ai_message:
                logger.info("DynamicAI generated an empty response. Skipping LM Studio turn.")
                conversation_log.append("DynamicAI: [No response generated. Still learning...]")
                yield "\n\n".join(conversation_log)
                time.sleep(delay)
                continue

            lm_studio_response = self.send_to_lm_studio(ai_message)
            if lm_studio_response:
                logger.info(f"LM Studio: {lm_studio_response}")
                conversation_log.append(f"LM Studio: {lm_studio_response}")
                message = lm_studio_response
                yield "\n\n".join(conversation_log)
            else:
                logger.warning("No response from LM Studio. Ending conversation.")
                break

            time.sleep(delay)

    def send_to_lm_studio(self, message):
        if not message.strip():
            logger.warning("Attempted to send an empty message to LM Studio. Skipping.")
            return None

        try:
            completion = self.lm_studio_client.ChatCompletion.create( 
                # You may need to change the model name here especially if you run multiple AI's in LM Studio
                model="unsloth/Llama-2-7b-Chat-hf",
                messages=[
                    {"role": "system", "content": "You're talking to an experimental AI that is still learning to communicate. If it doesn't respond or sends empty messages, please be patient and continue the conversation."},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
            )
            response = completion.choices[0].message.content
            return response
        except Exception as e:
            logger.error(f"Error sending to LM Studio: {str(e)}")
            return None

# -----------------------------
# 2. Additional Enhanced Components
# -----------------------------

### 2.1 LongTermMemory
class LongTermMemory(nn.Module):
    def __init__(self, memory_size, key_size, value_size):
        super(LongTermMemory, self).__init__()
        self.memory_size = memory_size
        self.key_size = key_size
        self.value_size = value_size
        self.memory = nn.Parameter(torch.randn(memory_size, key_size + value_size))
        self.current_index = 0

    def store(self, key, value):
        # Simple FIFO memory storage
        if self.current_index >= self.memory_size:
            self.current_index = 0
        with torch.no_grad():
            self.memory[self.current_index] = torch.cat((key, value), dim=1).squeeze(0)
            self.current_index += 1

    def retrieve(self, query):
        # Simple retrieval based on cosine similarity
        keys = self.memory[:, :self.key_size]  # (memory_size, key_size)
        values = self.memory[:, self.key_size:]  # (memory_size, value_size)
        similarities = F.cosine_similarity(query.unsqueeze(1), keys.unsqueeze(0), dim=2)  # (batch_size, memory_size)
        topk = torch.topk(similarities, k=1, dim=1)
        retrieved_values = values[topk.indices.squeeze(1)]  # (batch_size, value_size)
        return retrieved_values

### 2.2 AdaptiveLearningRate
class AdaptiveLearningRate:
    def __init__(self, optimizer, initial_lr=0.001, min_lr=1e-6, max_lr=0.1):
        self.optimizer = optimizer
        self.lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr

    def adjust(self, performance_metric, novelty_score):
        # Simple heuristic: Increase lr if performance improves and novelty is high
        if performance_metric < 0.5 and novelty_score > 0.5:
            self.lr = min(self.lr * 1.05, self.max_lr)
        else:
            self.lr = max(self.lr * 0.95, self.min_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

### 2.3 ExplainabilityModule
class ExplainabilityModule:
    def __init__(self, brain_model):
        self.brain = brain_model

    def generate_explanation(self, input_data, response):
        """
        Generate an explanation based on the model's internal activations.
        """
        # Example: Identify top contributing neurons in the root fractal neuron
        if len(self.brain.root.child_neurons) == 0:
            top_neurons = []
        else:
            activities = [child.soma.activity for child in self.brain.root.child_neurons]
            top_indices = torch.topk(torch.tensor(activities), min(5, len(activities))).indices.tolist()
            top_neurons = top_indices

        explanation = f"The response was influenced primarily by neurons {top_neurons} which detected key features in the input."
        return explanation

    def visualize_neuron_activations(self, input_data):
        # Placeholder for visualization logic
        pass

### 2.4 FractalRegularization
class FractalRegularization(nn.Module):
    def __init__(self, lambda_param=0.01):
        super(FractalRegularization, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, model):
        reg_loss = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                reg_loss += torch.sum(torch.abs(param))
        return self.lambda_param * reg_loss

### 2.5 MultiModalEncoder
class MultiModalEncoder(nn.Module):
    def __init__(self, text_dim, image_dim, audio_dim, output_dim):
        super(MultiModalEncoder, self).__init__()
        self.text_encoder = nn.Linear(text_dim, output_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.audio_encoder = nn.Linear(audio_dim, output_dim)
        # Assuming image_dim is square and divisible by 2 after pooling
        self.fc = nn.Linear(16 * (image_dim // 2) * (image_dim // 2) + output_dim * 2, output_dim)

    def forward(self, text_input, image_input, audio_input):
        text_encoded = self.text_encoder(text_input)  # (batch_size, output_dim)
        image_encoded = self.image_encoder(image_input)  # (batch_size, 16, H/2, W/2)
        image_encoded = image_encoded.view(image_encoded.size(0), -1)  # (batch_size, 16*(H/2)*(W/2))
        audio_encoded = self.audio_encoder(audio_input)  # (batch_size, output_dim)
        combined = torch.cat((text_encoded, image_encoded, audio_encoded), dim=1)  # (batch_size, ...)
        output = self.fc(combined)  # (batch_size, output_dim)
        return output

### 2.6 ContinuousLearner
class ContinuousLearner:
    def __init__(self, model, buffer_size=1000):
        self.model = model
        self.experience_buffer = []
        self.buffer_size = buffer_size

    def update(self, new_data):
        # Implement model update with new_data
        # Placeholder: Add new_data to buffer and sample from it
        self.add_to_buffer(new_data)
        if len(self.experience_buffer) >= self.buffer_size:
            sampled_data = random.sample(self.experience_buffer, self.buffer_size // 2)
            # Implement training on sampled_data
            pass

    def add_to_buffer(self, experience):
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)

# -----------------------------
# 3. Core Models
# -----------------------------

### 3.1 VAE Class for Latent Space Encoding
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        z = self.sample_latent(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def sample_latent(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss()(recon_x, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

# -----------------------------
# 5. DynamicAI Class with All Enhancements
# -----------------------------

class DynamicAI(nn.Module):
    def __init__(
        self,
        vocab_size=100000,
        embed_dim=300,
        latent_dim=300,
        output_dim=300,
        max_depth=14,
        num_intents=10,
        num_entities=10
    ):
        super(DynamicAI, self).__init__()
        self.embed_dim = embed_dim  # Store embed_dim as an instance variable

        # External API Client (Assuming OpenAI's API for LM Studio)
        self.lm_studio_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        # Initialize Tokenizers
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Add special tokens to BERT and GPT-2 tokenizers, including '[PAD]'
        self.additional_special_tokens = ["<USER>", "<NAME>", "<EMAIL>", "<PASSWORD>", "[PAD]"]
        self.bert_tokenizer.add_special_tokens({'additional_special_tokens': self.additional_special_tokens})
        self.gpt_tokenizer.add_special_tokens({'additional_special_tokens': self.additional_special_tokens})

        # Set the pad_token for GPT-2 tokenizer
        self.gpt_tokenizer.pad_token = '[PAD]'

        # Initialize Embeddings and Models
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.vae = VAE(embed_dim, latent_dim)
        self.brain = FractalBrain(latent_dim, output_dim, max_depth=max_depth)  # Updated input_dim to latent_dim
        self.wernickes = WernickesModule(num_intents=num_intents, num_entities=num_entities)
        self.brocas = BrocasModule()

        # Resize embeddings to accommodate new tokens
        self.wernickes.bert.resize_token_embeddings(len(self.bert_tokenizer))
        self.brocas.gpt.resize_token_embeddings(len(self.gpt_tokenizer))

        # Initialize Optimizer and Scheduler
        self.optimizer = optim.Adam(
            list(self.vae.parameters()) +
            list(self.brain.parameters()) +
            list(self.word_embeddings.parameters()) +
            list(self.wernickes.parameters()) +
            list(self.brocas.parameters()),
            lr=0.0001
        )
        self.adaptive_lr = AdaptiveLearningRate(self.optimizer, initial_lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.criterion = nn.MSELoss()

        # Vocabulary
        self.word_to_index = {}
        self.index_to_word = {}
        self.next_index = 0

        # Initialize Long-Term Memory
        self.long_term_memory = LongTermMemory(memory_size=1000, key_size=latent_dim, value_size=output_dim)

        # Initialize Explainability Module
        self.explainability = ExplainabilityModule(self.brain)

        # Initialize Continuous Learner
        self.continuous_learner = ContinuousLearner(self.brain, buffer_size=1000)

    def build_vocabulary(self, corpus, max_vocab_size=30000):
        """
        Builds the vocabulary using the Tokenizers library for efficiency.

        Args:
            corpus (list of str): List of sentences to build the vocabulary from.
            max_vocab_size (int): Maximum size of the vocabulary.
        """
        tokenizer_dir = "tokenizer"

        if os.path.exists(tokenizer_dir) and os.listdir(tokenizer_dir):
            logger.info("Loading existing tokenizer from directory...")
            # Load the tokenizer using transformers
            self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            # Add special tokens to GPT-2 tokenizer
            self.gpt_tokenizer.add_special_tokens({'additional_special_tokens': self.additional_special_tokens})
            self.gpt_tokenizer.pad_token = '[PAD]'
        else:
            logger.info("Training new tokenizer...")
            # Initialize the tokenizer
            tokenizer = BertWordPieceTokenizer(lowercase=True)

            # Train the tokenizer on the corpus
            tokenizer.train_from_iterator(
                corpus,
                vocab_size=max_vocab_size,
                special_tokens=[
                    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
                ] + self.additional_special_tokens  # Include additional special tokens
            )

            # Save the tokenizer to a directory
            tokenizer.save_model(tokenizer_dir)

            # Load the tokenizer using transformers
            self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)

            # Add special tokens to BERT tokenizer (again to ensure consistency)
            self.bert_tokenizer.add_special_tokens({'additional_special_tokens': self.additional_special_tokens})

            # Load GPT-2 tokenizer and add special tokens
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.gpt_tokenizer.add_special_tokens({'additional_special_tokens': self.additional_special_tokens})
            self.gpt_tokenizer.pad_token = '[PAD]'

        # Update the vocabulary size
        self.vocab_size = len(self.bert_tokenizer)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_dim)

        # Build word_to_index and index_to_word
        self.word_to_index = self.bert_tokenizer.get_vocab()
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}

        # Resize embeddings in models
        self.wernickes.bert.resize_token_embeddings(self.vocab_size)
        self.brocas.gpt.resize_token_embeddings(len(self.gpt_tokenizer))

        logger.info(f"Vocabulary built with {self.vocab_size} tokens.")

    def initialize_weights(self):
        """
        Initialize model weights using Xavier Uniform for linear layers and Kaiming Uniform for convolutional layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def tokenize_input(self, text):
        """
        Tokenize input text using BERT tokenizer for comprehension.
        """
        tokens = self.bert_tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        return tokens['input_ids'], tokens['attention_mask']

    def tokenize_response(self, text):
        """
        Tokenize response text using GPT-2 tokenizer for production.
        """
        tokens = self.gpt_tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        return tokens['input_ids'], tokens['attention_mask']

    def embedding_to_text(self, embedding):
        """
        Convert an embedding vector to the closest word in the embedding space.
        """
        with torch.no_grad():
            # Reshape if necessary
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            # Normalize the embedding
            embedding_norm = F.normalize(embedding, p=2, dim=1)  # Shape: (batch_size, embed_dim)

            # Normalize all word embeddings
            word_embeddings_norm = F.normalize(self.word_embeddings.weight, p=2, dim=1)  # Shape: (vocab_size, embed_dim)

            # Compute cosine similarity
            similarities = torch.matmul(embedding_norm, word_embeddings_norm.t())  # Shape: (batch_size, vocab_size)

            # Get the index of the highest similarity
            best_match_indices = torch.argmax(similarities, dim=1)  # Shape: (batch_size,)

            # Retrieve the corresponding words
            words = [self.index_to_word.get(idx.item(), "[UNKNOWN]") for idx in best_match_indices]

            return ' '.join(words)

    def chat(self, input_text, max_new_tokens=50, temperature=0.7):
        # Tokenize input using BERT tokenizer
        input_ids, attention_mask = self.tokenize_input(input_text)

        thinking_process = []
        with torch.no_grad():
            # Wernicke's Module: Comprehension
            intent_logits, entity_logits = self.wernickes(input_ids, attention_mask)
            intent = torch.argmax(intent_logits, dim=1)
            entities = torch.argmax(entity_logits, dim=2)
            thinking_process.append(f"Intent: {intent.tolist()}, Entities: {entities.tolist()}")

            # Convert input to embeddings
            embedded_q = self.word_embeddings(input_ids)
            embedded_q = embedded_q.mean(dim=1)

            # VAE Encoding
            _, _, _, z_q = self.vae(embedded_q)

            # Brain Processing
            brain_output = self.brain.forward(z_q)

            # VAE Decoding
            decoded_output = self.vae.decoder(brain_output)
            thinking_process.append(f"Decoded Output Vector: {decoded_output.tolist()}")

        # Convert decoded output to text to use as GPT-2's prompt
        prompt_text = self.embedding_to_text(decoded_output)
        prompt_ids, prompt_attention_mask = self.tokenize_response(prompt_text)

        # Broca's Module: Production using the prompt
        with torch.no_grad():
            response_ids = self.brocas(
                prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            response_text = self.gpt_tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Explainability
        explanation = self.explainability.generate_explanation(input_text, response_text)

        # Construct the final response
        thinking_str = "\n".join(thinking_process)
        final_response = (
            f"Thinking Process:\n{thinking_str}\n\n"
            f"Explanation:\n{explanation}\n\n"
            f"Response: {response_text}"
        )

        return final_response

    def save_state(self, filename):
        state = {
            'word_embeddings_state': self.word_embeddings.state_dict(),
            'vae_state': self.vae.state_dict(),
            'brain_state': self.brain.state_dict(),
            'wernickes_state': self.wernickes.state_dict(),
            'brocas_state': self.brocas.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'word_to_index': self.word_to_index,
            'index_to_word': self.index_to_word,
            'next_index': self.next_index,
            'long_term_memory': self.long_term_memory.memory.data
        }
        torch.save(state, filename)
        logger.info(f"Model state saved to {filename}")

    def load_state(self, filename):
        state = torch.load(filename)
        self.word_to_index = state['word_to_index']
        self.index_to_word = state['index_to_word']
        self.next_index = state['next_index']
        self.word_embeddings.load_state_dict(state['word_embeddings_state'])
        self.vae.load_state_dict(state['vae_state'])
        self.brain.load_state_dict(state['brain_state'])
        self.wernickes.load_state_dict(state['wernickes_state'])
        self.brocas.load_state_dict(state['brocas_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
        self.long_term_memory.memory.data = state.get(
            'long_term_memory',
            torch.randn(1000, self.brain.root.soma.weights.size(1) + self.brain.root.soma.weights.size(1))
        )
        logger.info(f"Model state loaded from {filename}")

    def train_on_qa_pairs(self, qa_pairs, epochs=10):
        if not isinstance(qa_pairs, list) or len(qa_pairs) == 0:
            raise ValueError("qa_pairs must be a non-empty list")

        logger.info(f"Training on {len(qa_pairs)} Q&A pairs for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0
            errors = 0
            random.shuffle(qa_pairs)
            for i, (question, answer) in enumerate(qa_pairs):
                self.optimizer.zero_grad()

                try:
                    # Tokenize question and answer
                    q_input_ids, q_attention_mask = self.tokenize_input(question)
                    a_input_ids, a_attention_mask = self.tokenize_input(answer)

                    # Wernicke's Comprehension for Question
                    q_intent_logits, q_entity_logits = self.wernickes(q_input_ids, q_attention_mask)
                    q_intent = torch.argmax(q_intent_logits, dim=1)
                    q_entities = torch.argmax(q_entity_logits, dim=2)

                    # Convert question to embeddings
                    q_embedded = self.word_embeddings(q_input_ids)
                    q_embedded = q_embedded.mean(dim=1)

                    # VAE Encoding and Decoding for Question
                    q_recon, q_mu, q_logvar, z_q = self.vae(q_embedded)
                    q_latent_output = self.brain.forward(z_q)
                    q_decoded_output = self.vae.decoder(q_latent_output)

                    # VAE Loss for Question
                    vae_loss_q = self.vae.vae_loss(q_recon, q_embedded, q_mu, q_logvar)

                    # Convert answer to embeddings
                    a_embedded = self.word_embeddings(a_input_ids)
                    a_embedded = a_embedded.mean(dim=1)

                    # VAE Encoding and Decoding for Answer
                    a_recon, a_mu, a_logvar, a_latent = self.vae(a_embedded)
                    a_latent_output = self.brain.forward(a_latent)
                    a_decoded_output = self.vae.decoder(a_latent_output)

                    # VAE Loss for Answer
                    vae_loss_a = self.vae.vae_loss(a_recon, a_embedded, a_mu, a_logvar)

                    # Calculate Loss
                    loss = self.criterion(q_decoded_output, a_decoded_output) + vae_loss_q + vae_loss_a
                    loss += FractalRegularization(lambda_param=0.01)(self.brain)
                    loss.backward()

                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.word_embeddings.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.wernickes.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.brocas.parameters(), max_norm=1.0)

                    # Update Parameters
                    self.optimizer.step()

                    total_loss += loss.item()

                    # Dynamic Growth and Pruning
                    self.brain.grow()
                    self.brain.prune()
                    self.brain.emotional_module.update_emotion(torch.mean(loss).item())
                    self.brain.curiosity_module.update(z_q)
                    self.long_term_memory.store(z_q, a_latent)
                    self.continuous_learner.update((z_q, a_latent))

                    if i % 10 == 0:
                        logger.info(f"Epoch {epoch+1}, Pair {i+1}/{len(qa_pairs)}, Loss: {loss.item():.4f}")

                except Exception as e:
                    logger.error(f"Error processing pair: {question} | {answer}")
                    logger.error(f"Error details: {str(e)}")
                    errors += 1
                    continue

            avg_loss = total_loss / (len(qa_pairs) - errors) if len(qa_pairs) > errors else 0
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Errors: {errors}")

            # Adjust learning rate based on performance
            novelty = self.brain.curiosity_module.evaluate_novelty(z_q)
            self.adaptive_lr.adjust(performance_metric=avg_loss, novelty_score=novelty)

            self.scheduler.step()
            self.save_state(f"model_state_epoch_{epoch+1}.pth")

            yield epoch + 1, avg_loss, errors

    def think_mode(self, max_iterations=10):
        """
        Simulate the AI's internal thought process over multiple iterations without external input or output.
        """
        # Initialize a random latent thought vector
        thought = torch.randn(1, self.vae.latent_dim)
    
        for iteration in range(max_iterations):
            with torch.no_grad():
                # Brain Processing
                brain_output = self.brain.forward(thought)
            
                # VAE Decoding and Re-encoding
                decoded_output = self.vae.decoder(brain_output)
                _, _, _, new_thought = self.vae(decoded_output)
        
            # Update thought for next iteration
            thought = new_thought

    def talk_with_lm_studio(self, initial_message, conversation_duration=60, delay=2):
        message = initial_message
        start_time = time.time()
        conversation_log = []

        while time.time() - start_time < conversation_duration:
            ai_response = self.chat(message)
            logger.info(f"DynamicAI:\n{ai_response}")
            conversation_log.append(f"DynamicAI:\n{ai_response}")
            yield "\n\n".join(conversation_log)

            ai_message = ai_response.split("Response: ")[-1].strip()

            if not ai_message:
                logger.info("DynamicAI generated an empty response. Skipping LM Studio turn.")
                conversation_log.append("DynamicAI: [No response generated. Still learning...]")
                yield "\n\n".join(conversation_log)
                time.sleep(delay)
                continue

            lm_studio_response = self.send_to_lm_studio(ai_message)
            if lm_studio_response:
                logger.info(f"LM Studio: {lm_studio_response}")
                conversation_log.append(f"LM Studio: {lm_studio_response}")
                message = lm_studio_response
                yield "\n\n".join(conversation_log)
            else:
                logger.warning("No response from LM Studio. Ending conversation.")
                break

            time.sleep(delay)

    def send_to_lm_studio(self, message):
        if not message.strip():
            logger.warning("Attempted to send an empty message to LM Studio. Skipping.")
            return None

        try:
            # Correct API call for LM Studio
            completion = self.lm_studio_client.chat.completions.create(
                model="unsloth/Llama-2-7b-Chat-hf",  # Replace with your LM Studio model name
                messages=[
                    {"role": "system", "content": "You're talking to an experimental AI that is still learning to communicate. If it doesn't respond or sends empty messages, please be patient and continue the conversation."},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
            )
            response = completion.choices[0].message.content
            return response
        except Exception as e:
            logger.error(f"Error sending to LM Studio: {str(e)}")
            return None
        
# -----------------------------
# 6. Gradio Interface
# -----------------------------

def create_gradio_interface(ai):
    def handle_chat(message, temperature):
        return ai.chat(message, max_new_tokens=50, temperature=float(temperature))
    
    def handle_save(filename):
        ai.save_state(filename)
        return f"State saved to {filename}"
    
    def handle_load(filename):
        ai.load_state(filename)
        return f"State loaded from {filename}"
    
    def handle_train_qa(qa_pairs_file, epochs):
        try:
            with open(qa_pairs_file.name, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)

            output = ["Starting training..."]
            for epoch, loss, errors in ai.train_on_qa_pairs(qa_pairs, epochs=int(epochs)):
                output.append(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Errors: {errors}")
            output.append("Training completed successfully")
            return "\n".join(output)
        except Exception as e:
            return f"Error during training: {str(e)}"
    
    def handle_lm_studio_chat(initial_message, duration, delay):
        conversation_log = ""
        for log in ai.talk_with_lm_studio(initial_message, conversation_duration=float(duration), delay=float(delay)):
            conversation_log = log
            yield conversation_log

    def handle_think_mode(max_iterations):
        ai.think_mode(max_iterations=int(max_iterations))
        return "Think mode completed internally."

    with gr.Blocks() as interface:
        gr.Markdown("# Dynamic AI with Enhanced Fractal Neuron Model")

        with gr.Tab("Chat"):
            chat_input = gr.Textbox(label="Your message")
            temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature")
            chat_output = gr.Textbox(label="AI response")
            chat_button = gr.Button("Chat")
            chat_button.click(handle_chat, inputs=[chat_input, temperature], outputs=chat_output)

        with gr.Tab("LM Studio Conversation"):
            initial_message = gr.Textbox(label="Initial Message")
            duration = gr.Number(label="Conversation Duration (seconds)", value=60)
            delay = gr.Number(label="Delay between messages (seconds)", value=2)
            conversation_log = gr.Textbox(label="Conversation Log", lines=20)
            start_conversation = gr.Button("Start Conversation")
            start_conversation.click(handle_lm_studio_chat, inputs=[initial_message, duration, delay], outputs=conversation_log)

        with gr.Tab("Think Mode"):
            max_iterations = gr.Number(label="Max Iterations", value=10)
            thought_log = gr.Textbox(label="Thought Log", lines=20)
            start_thinking = gr.Button("Start Thinking")
            stop_thinking = gr.Button("Stop Thinking")

            # Event handling for Think Mode
            think_event = start_thinking.click(
                handle_think_mode,
                inputs=[max_iterations],
                outputs=thought_log
            )
            stop_thinking.click(lambda: None, cancels=[think_event])

        with gr.Tab("Train on Q&A"):
            qa_file = gr.File(label="Q&A Pairs JSON File")
            epochs_input = gr.Number(label="Number of Epochs", value=10)
            train_button = gr.Button("Train on Q&A Pairs")
            train_output = gr.Textbox(label="Training status")
            train_button.click(handle_train_qa, inputs=[qa_file, epochs_input], outputs=train_output)

        with gr.Tab("Save/Load State"):
            filename_input = gr.Textbox(label="Filename")
            save_button = gr.Button("Save State")
            load_button = gr.Button("Load State")
            state_output = gr.Textbox(label="Operation result")
            save_button.click(handle_save, inputs=[filename_input], outputs=state_output)
            load_button.click(handle_load, inputs=[filename_input], outputs=state_output)

    return interface

# -----------------------------
# 7. Main Execution
# -----------------------------

def load_pretrained_embeddings(dynamic_ai, embedding_dim=300):
    # Load GloVe embeddings
    glove = GloVe(name='6B', dim=embedding_dim)
    
    # Assign pre-trained weights to word_embeddings
    for word, idx in dynamic_ai.word_to_index.items():
        if word in glove.stoi:
            dynamic_ai.word_embeddings.weight.data[idx] = glove.vectors[glove.stoi[word]]
        else:
            # Initialize embeddings for unknown words
            dynamic_ai.word_embeddings.weight.data[idx] = torch.randn(embedding_dim)
    
    logger.info("Pre-trained embeddings loaded successfully.")

def load_large_corpus(cache_dir='./cache', corpus_file='wikitext103_corpus.pkl'):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    corpus_path = os.path.join(cache_dir, corpus_file)

    if os.path.exists(corpus_path):
        logger.info("Loading WikiText103 corpus from local cache...")
        with open(corpus_path, 'rb') as f:
            corpus = pickle.load(f)
    else:
        logger.info("Downloading and loading WikiText103 corpus...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split='train', cache_dir=cache_dir)
        corpus = [item['text'] for item in dataset if item['text'].strip()]
        # Save the corpus to a local file for future use
        with open(corpus_path, 'wb') as f:
            pickle.dump(corpus, f)
        logger.info(f"Corpus saved to {corpus_path}")

    logger.info(f"Loaded {len(corpus)} sentences from WikiText103.")
    return corpus

if __name__ == "__main__":
    dynamic_ai = DynamicAI(
        vocab_size=100000,
        embed_dim=300,  # Ensure this matches the GloVe embedding dimension
        latent_dim=300,
        output_dim=300,
        max_depth=14,
        num_intents=10,
        num_entities=10
    )

    # Initialize model weights properly
    dynamic_ai.initialize_weights()

    # Check if the tokenizer needs to be built
    large_corpus = load_large_corpus()
    dynamic_ai.build_vocabulary(large_corpus)

    # Load pre-trained embeddings
    load_pretrained_embeddings(dynamic_ai, embedding_dim=300)

    # Optionally, print the vocabulary size
    print(f"Vocabulary Size: {len(dynamic_ai.index_to_word)}")

    # Create and launch the Gradio interface
    iface = create_gradio_interface(dynamic_ai)
    iface.launch()
