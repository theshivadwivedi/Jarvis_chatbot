# LSTM + Luong Attention Chatbot

A simple conversational chatbot built using TensorFlow (Seq2Seq with Luong Attention) and served with FastAPI.  
Includes a clean web-based chat UI built using HTML, CSS, and JavaScript.

---

## Overview

This project uses:

- Bidirectional LSTM Encoder  
- LSTM Decoder  
- Luong Dot Attention  
- Greedy decoding at inference  
- FastAPI backend  
- Simple modern frontend UI (ChatGPT style)

The chatbot is trained on custom Q&A pairs and generates responses based on learned patterns.

---

## Project Structure

project/
│ app.py                   → FastAPI API server  
│ inference_engine.py      → Loads model and generates replies  
│ requirements.txt         → Python dependencies  
│ README.md  
│
├── static/                → Frontend files  
│     index.html  
│     style.css  
│     app.js  
│
└── model/                 → Place model files here  
      seq2seq_luong_glove_final.h5  
      tokenizer.pkl  
      prep_meta.json  

---

## Model Files (Required)

Place the following files inside the **model/** folder:

- seq2seq_luong_glove_final.h5  
- tokenizer.pkl  
- prep_meta.json  

These files are **not included in the repository** due to size limits.

---


