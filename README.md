# Neural Networks: Zero to Hero 🚀

This repository contains my implementations and notes from Andrej Karpathy’s *Neural Networks: Zero to Hero* lecture series.  
The goal is to deeply understand how modern neural networks and Large Language Models work — by building everything from scratch, step by step.

---

## 📚 Overview

The series covers the full journey from the foundations of automatic differentiation to building a GPT-style transformer model.

| Module | Description |
|--------|-------------|
| **micrograd/** | A tiny autograd engine and a small neural net library built from scratch. |
| **makemore/** | Character-level language models (bigram, MLP, and simple neural nets). |
| **nanogpt/** | An implementation of the decoder only architecture based on "Attention is All you Need". |
| **tokenizer/** | Tokenization utilities: encode/decode logic, vocabulary building, and preprocessing. |
| **gpt2/** | A GPT-style transformer implementation with attention, layer normalization, and training loops. |

---

## 🧠 Learning Goals
- Strengthen intuition for how backpropagation and gradient descent really work  
- Understand neural network design from first principles  
- Write core ML components (forward, backward, training) without frameworks  
- Connect these fundamentals to how modern LLMs like GPT are built

---

## 🛠️ Tech Stack
- Python  
- PyTorch (for parts where abstraction helps)  
- NumPy  
- Jupyter Notebooks  

---

## 📝 TODO
- Move code from notebooks into cleaner executable modules (`src/` structure)  
- Complete all side missions and bonus exercises suggested by Karpathy  

---

## 🙌 Acknowledgments
All credits to **Andrej Karpathy** for the amazing *Zero to Hero* series.  
I’m simply reimplementing and documenting my learning journey here.

---

> “What I cannot create, I do not understand.” — Richard Feynman
