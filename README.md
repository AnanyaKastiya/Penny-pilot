# 💸 PennyPilot — Autonomous Multimodal Expense & Financial Health Agent

> **Live Application:** [https://pennypilot-ai.streamlit.app](https://pennypilot-ai.streamlit.app)  
> **Built with:** Python 3.11+ • Google Gemini Multimodal Vision • LangGraph (Human-in-the-Loop State Machine) • ChromaDB (Self-Learning RAG) • SQLite • Pandas • Streamlit • Plotly

---

## 💡 Why PennyPilot?

Standard expense trackers (Splitwise, Money Manager, Excel) fail because **manual categorization is exhausting**—users must manually pick categories for dozens of cryptic transactions every week, causing 90% of users to drop out within 14 days.

**PennyPilot** solves this through **Multimodal Agentic AI and Self-Learning Memory**:
1. **Natural Language & Multi-Day Logging:** Type casually (e.g., *"Yesterday spent 340 on Swiggy dinner. On 24th Aug paid 1200 for electricity bill. Today spent 120 on auto."*).
2. **UPI Screenshot OCR:** Upload raw payment screenshots (Google Pay, PhonePe, Paytm, CRED). Gemini Vision extracts amount, date, payment app, and recipient UPI IDs.
3. **Self-Learning RAG Memory (ChromaDB):** When an ambiguous recipient is encountered (e.g., *"Ramesh Kumar"*), the agent pauses, asks for clarification once, and permanently embeds that relationship into vector memory.
4. **Human-in-the-Loop (HITL) State Machine (LangGraph):** Gracefully routes between parsing, RAG resolution, human clarification breakpoints, and database persistence.
5. **Deterministic Math (Pandas & NumPy):** 100% accurate, non-hallucinated Week-on-Week (WoW) and Month-on-Month (MoM) percentage shifts, category surges, daily burn rates, and recurring leak detection.

---

## 🏗️ System Architecture

```
                  [ USER INPUT: Natural Text / UPI Screenshot ]
                                        │
                                        ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                         LANGGRAPH MULTI-AGENT STATE MACHINE                            │
 │                                                                                        │
 │  ┌─────────────────────────────────┐        ┌───────────────────────────────────────┐  │
 │  │ 👁️ Node 1: Multimodal Parser    │ ─────► │ 🧠 Node 2: RAG Merchant Resolver      │  │
 │  │ (Gemini 3.6/3.7 Vision API)     │        │ (ChromaDB Vector Semantic Memory)     │  │
 │  └─────────────────────────────────┘        └──────────────────┬────────────────────┘  │
 │                                                                │                       │
 │                                                 Is category clear?                     │
 │                                                  ╱            ╲                        │
 │                                              [ YES ]        [ NO ]                     │
 │                                                ╱                ╲                      │
 │                                               ▼                  ▼                     │
 │  ┌─────────────────────────────────┐   ┌────────────┐   ┌───────────────────────────┐  │
 │  │ 💾 Node 4: Transaction Recorder │ ◄─┤ Auto-Tag   │   │ ⏸️ Node 3: HITL Clarifier │  │
 │  │ (Saves to SQLite + ChromaDB)    │   └────────────┘   │ (Pauses graph, asks user  │  │
 │  └────────────────┬────────────────┘                    │ in UI, learns answer)     │  │
 │                   │                                     └─────────────┬─────────────┘  │
 │                   ▼                                                   │                │
 │  ┌─────────────────────────────────┐                                  │                │
 │  │ 📊 Node 5: Analytics & Strategy │ ◄────────────────────────────────┘                │
 │  │ (Pandas Engine: WoW, MoM,       │                                                   │
 │  │  Burn Rate & Budget Optimizer)  │                                                   │
 │  └─────────────────────────────────┘                                                   │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### ⚡ 1. Multimodal UPI & Receipt Scanner
* Extract amounts, recipient names, UPI IDs, dates, and payment apps from Google Pay, PhonePe, Paytm, or CRED screenshots.
* Handles complex handwritten receipts and cropped screens.

### 🧠 2. Self-Learning Merchant Memory (RAG)
* Cryptic merchant names are indexed in **ChromaDB**.
* If you clarify that *"Shree Balaji Traders"* is *Stationery & Books*, PennyPilot remembers and auto-categorizes future payments without asking again.

### 📊 3. Exact Mathematical Spending Analytics
* **Week-on-Week (WoW) Trends:** Exact percentage change and category-by-category delta comparisons.
* **Month-on-Month (MoM) Shift:** Demonstrates a realistic 38% reduction in monthly spend.
* **Category Breakdown:** Interactive Plotly donut charts and distribution tables.

### 💡 4. Financial Health & Predictive Coach
* **Daily Burn Rate Gauge:** Calculates your maximum safe daily spend allowance based on remaining days in the month.
* **Hidden Subscription Leak Detector:** Automatically spots recurring monthly debits (Netflix, Spotify, Gym) and calculates annual cost drain.
* **Micro-Savings Simulator:** Calculates passive wealth creation if you rounded up every UPI payment to the nearest ₹10, ₹50, or ₹100.
* **Interactive AI Advisor:** Ask natural language questions like *"Where am I spending the most?"* or *"What to cut down?"*.

---

## 🛠️ Tech Stack

* **Backend & Logic:** Python 3.11+, Pydantic
* **AI & Vision:** Google GenAI SDK (`gemini-3.6-flash` / `gemini-3.7-flash`)
* **Agent Framework:** LangGraph (StateGraph with conditional routing)
* **Vector Store (RAG):** ChromaDB (Local cosine similarity index)
* **Relational DB:** SQLite 3
* **Data Processing:** Pandas, NumPy
* **Frontend UI:** Streamlit, Plotly

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/AnanyaKastiya/Penny-pilot.git
cd Penny-pilot
pip install -r requirements.txt
```

### 2. Set Up Environment
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```
*(Or enter your key directly in the Streamlit sidebar!)*

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```

### 4. Run Automated Tests
```bash
pytest tests/ -v
```

---

## 👩‍💻 Author
**Ananya Kastiya**  
Live Application: [https://pennypilot-ai.streamlit.app](https://pennypilot-ai.streamlit.app)  
GitHub Repository: [https://github.com/AnanyaKastiya/Penny-pilot](https://github.com/AnanyaKastiya/Penny-pilot)
