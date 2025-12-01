# Sample Prompts for Qwen3-1.8B Base Model

This is a **base model** (not instruction-tuned), so **completion-style prompts** work best.
Start a sentence and let the model continue it.

---

## 🌍 Factual Completion

```
The capital of France is
```

```
Albert Einstein was born in
```

```
The largest planet in our solar system is
```

```
Water boils at a temperature of
```

```
The speed of light is approximately
```

```
The Great Wall of China was built during the
```

---

## 💻 Code Completion

```
# Python function to calculate factorial
def factorial(n):
```

```
# Python function to check if a number is prime
def is_prime(n):
```

```
# Python: Read a JSON file and print its contents
import json

with open('data.json', 'r') as f:
```

```
# Python: Make an HTTP GET request
import requests

response = requests.get('https://api.example.com/data')
```

```
# Python: Sort a list of dictionaries by a key
data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
sorted_data =
```

```
// JavaScript function to reverse a string
function reverseString(str) {
```

---

## 🧠 Technical/ML Concepts

```
In machine learning, gradient descent is used to
```

```
The difference between supervised and unsupervised learning is that
```

```
A neural network consists of
```

```
The attention mechanism in transformers allows the model to
```

```
Backpropagation is an algorithm that
```

```
The purpose of dropout in neural networks is to
```

---

## ✨ Creative Writing

```
Once upon a time, in a kingdom far away,
```

```
The old lighthouse stood at the edge of the cliff,
```

```
She opened the letter and read the first line:
```

```
The spaceship landed on the barren planet, and the crew
```

```
In the year 2150, humanity had finally
```

---

## 📝 Text Continuation

```
The three main branches of the United States government are
```

```
The process of photosynthesis involves
```

```
To make a cup of coffee, first you need to
```

```
The main difference between TCP and UDP is that TCP
```

```
Object-oriented programming is based on the concept of
```

---

## 🔬 Scientific

```
DNA stands for deoxyribonucleic acid, and it is responsible for
```

```
The theory of relativity, proposed by Einstein, states that
```

```
Black holes are formed when
```

```
The periodic table organizes elements by their
```

---

## 💡 Recommended Settings

| Use Case | Temperature | Top K | Max Tokens |
|----------|-------------|-------|------------|
| Factual completion | 0.2-0.4 | 10-20 | 64 |
| Code completion | 0.2-0.3 | 5-15 | 256 |
| Technical explanation | 0.4-0.6 | 20-40 | 128 |
| Creative writing | 0.8-1.2 | 50-100 | 512 |
| Deterministic/greedy | 0.1 | 1 | varies |

---

## ⚠️ Prompts That Won't Work Well

These require instruction-tuning (SFT/RLHF):

```
❌ Write me a poem about the ocean
❌ Explain quantum computing to a 5 year old  
❌ What is the best programming language?
❌ Can you help me debug this code?
❌ Summarize this article for me
```

The model doesn't understand instructions – it only knows how to continue text.

