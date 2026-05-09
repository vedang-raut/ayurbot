"""
🌿 Prakriti Determine — Ayurvedic Body Type Chatbot
Built with Streamlit + scikit-learn Random Forest
"""

import streamlit as st
import numpy as np
import pickle
import os
import sys

# ─── Auto-train if model missing ──────────────────────────────────────────────
if not os.path.exists("model/prakriti_model.pkl"):
    with st.spinner("🌱 Training Prakriti model for the first time…"):
        import train_model
        train_model.train_and_save()

with open("model/prakriti_model.pkl", "rb") as f:
    MODEL = pickle.load(f)

# ─── Constants ────────────────────────────────────────────────────────────────
PRAKRITI_LABELS = ["Vata", "Pitta", "Kapha"]

PRAKRITI_INFO = {
    "Vata": {
        "emoji": "🌬️",
        "element": "Air & Space",
        "color": "#8B7FD4",
        "traits": "Creative, enthusiastic, quick-thinking, light, flexible",
        "qualities": "Dry, light, cold, rough, mobile, subtle",
        "description": (
            "Vata types are energetic and creative with quick minds. "
            "They tend to be slender, move fast, and think fast — but can become "
            "anxious or scattered when out of balance."
        ),
    },
    "Pitta": {
        "emoji": "🔥",
        "element": "Fire & Water",
        "color": "#E07B54",
        "traits": "Focused, determined, intelligent, organized, confident",
        "qualities": "Hot, sharp, light, oily, mobile, liquid",
        "description": (
            "Pitta types are natural leaders with sharp intellects and strong "
            "digestion. They are goal-oriented and passionate, but can become "
            "irritable or inflammatory when imbalanced."
        ),
    },
    "Kapha": {
        "emoji": "🌊",
        "element": "Earth & Water",
        "color": "#5B9E6E",
        "traits": "Calm, loving, patient, loyal, strong, nurturing",
        "qualities": "Heavy, slow, cold, oily, smooth, dense",
        "description": (
            "Kapha types are grounded, stable and nurturing with excellent stamina. "
            "They are loyal and compassionate, but can become lethargic or resistant "
            "to change when out of balance."
        ),
    },
}

DIET_SUGGESTIONS = {
    "Vata": {
        "principle": "Warm, moist, nourishing, and grounding foods",
        "favor": [
            "🍚 Warm cooked grains — rice, oatmeal, wheat",
            "🥣 Soups and stews with ghee or olive oil",
            "🧄 Root vegetables — carrots, beets, sweet potatoes",
            "🥛 Warm milk with turmeric and honey",
            "🫚 Healthy fats — avocado, sesame oil, ghee",
            "🍯 Sweet, sour, and salty tastes",
            "🌰 Soaked nuts — almonds, cashews, walnuts",
            "🍌 Sweet ripe fruits — bananas, mangoes, dates",
        ],
        "avoid": [
            "🧊 Cold, raw, or dry foods",
            "🥤 Carbonated and iced drinks",
            "🫘 Raw salads and beans (can cause gas)",
            "☕ Excess caffeine and alcohol",
            "🍿 Dry snacks — crackers, chips, popcorn",
        ],
        "meal_tips": [
            "🕗 Eat at regular, consistent times",
            "🍽️ Eat three warm meals a day",
            "🌡️ Favor warm beverages — herbal teas, warm water",
            "🧘 Eat in a calm, settled environment",
            "🥄 Add warming spices: ginger, cinnamon, cardamom, cumin",
        ],
        "herbs": "Ashwagandha, Shatavari, Triphala, Ginger, Licorice",
    },
    "Pitta": {
        "principle": "Cool, fresh, and moderately heavy foods",
        "favor": [
            "🥗 Fresh salads with cooling vegetables",
            "🥒 Cucumber, zucchini, leafy greens",
            "🍎 Sweet fruits — apples, pears, grapes, pomegranate",
            "🌾 Barley, oats, basmati rice",
            "🧀 Cooling dairy — milk, ghee, coconut milk",
            "🫛 Legumes — mung beans, lentils, chickpeas",
            "🌿 Sweet, bitter, and astringent tastes",
            "🥥 Coconut water and coconut oil",
        ],
        "avoid": [
            "🌶️ Spicy and pungent foods — chillies, hot peppers",
            "🍟 Fried and oily foods",
            "☕ Excess caffeine, alcohol, and sour drinks",
            "🍋 Fermented, sour foods — vinegar, pickles",
            "🍔 Red meat and heavy proteins",
        ],
        "meal_tips": [
            "🌙 Don't skip meals — Pitta hunger is intense",
            "❄️ Allow food to cool slightly before eating",
            "🌿 Add cooling herbs: coriander, fennel, mint, turmeric",
            "😌 Eat without rushing or multitasking",
            "🥤 Drink room-temperature or cool water, not ice cold",
        ],
        "herbs": "Brahmi, Amalaki, Neem, Coriander, Shatavari",
    },
    "Kapha": {
        "principle": "Light, warm, dry, and stimulating foods",
        "favor": [
            "🌶️ Spicy and pungent foods to stoke digestion",
            "🥬 Light leafy greens — spinach, kale, arugula",
            "🍎 Light fruits — apples, pears, pomegranate, berries",
            "🌾 Light grains — barley, millet, buckwheat, rye",
            "🫛 Legumes — all beans and lentils",
            "🧅 Pungent vegetables — onions, radishes, leeks",
            "🍯 Pungent, bitter, and astringent tastes",
            "☕ Warming herbal teas — ginger, black pepper, tulsi",
        ],
        "avoid": [
            "🍰 Heavy, sweet, and oily foods",
            "🥛 Excess dairy — cheese, yogurt, ice cream",
            "🍝 Refined carbs — white bread, pasta, pastries",
            "🍬 Sugar and sugary drinks",
            "🥩 Heavy meats and fatty foods",
        ],
        "meal_tips": [
            "🌅 Eat largest meal at midday when digestion is strongest",
            "🚶 Light exercise before meals to stimulate appetite",
            "🧂 Use pungent spices freely: black pepper, ginger, mustard",
            "🍽️ Eat two meals daily — avoid frequent snacking",
            "🌡️ Favor warm, lightly cooked, and dry preparations",
        ],
        "herbs": "Trikatu, Guggulu, Triphala, Punarnava, Tulsi",
    },
}

# ─── Questions ────────────────────────────────────────────────────────────────
QUESTIONS = [
    {
        "key": "body_frame",
        "text": "🏃 How would you describe your body frame?",
        "options": [
            "Slim, light, hard to gain weight",
            "Medium, muscular, athletic build",
            "Large, heavy, easy to gain weight",
        ],
    },
    {
        "key": "skin_type",
        "text": "✋ What is your skin type like?",
        "options": [
            "Dry, rough, or flaky skin",
            "Oily, sensitive, prone to rashes or redness",
            "Thick, smooth, well-moisturized",
        ],
    },
    {
        "key": "hair_type",
        "text": "💇 How would you describe your hair?",
        "options": [
            "Dry, frizzy, thin, or coarse",
            "Fine, straight, early greying or thinning",
            "Thick, wavy, lustrous, and oily",
        ],
    },
    {
        "key": "appetite",
        "text": "🍽️ How is your appetite?",
        "options": [
            "Variable — sometimes hungry, sometimes not",
            "Strong and sharp — very irritable when hungry",
            "Slow and steady — can skip meals easily",
        ],
    },
    {
        "key": "digestion",
        "text": "🫃 How is your digestion typically?",
        "options": [
            "Irregular — prone to gas, bloating, constipation",
            "Quick and strong — prone to acidity or heartburn",
            "Slow and sluggish — feel heavy after meals",
        ],
    },
    {
        "key": "sleep_pattern",
        "text": "😴 Describe your typical sleep:",
        "options": [
            "Light, interrupted, hard to fall asleep (6–7 hrs)",
            "Moderate and sound (7–8 hrs)",
            "Deep, long, and heavy — hard to wake up (8–10 hrs)",
        ],
    },
    {
        "key": "energy_level",
        "text": "⚡ What is your energy level like?",
        "options": [
            "Comes in bursts — high then crashes",
            "Intense and focused but can burn out",
            "Steady and enduring through the day",
        ],
    },
    {
        "key": "temperament",
        "text": "🧠 How would you describe your temperament?",
        "options": [
            "Creative, enthusiastic, but anxious or scattered",
            "Ambitious, decisive, but can get irritable or critical",
            "Calm, patient, and loving but can be stubborn",
        ],
    },
    {
        "key": "memory",
        "text": "🧩 How is your memory?",
        "options": [
            "Quick to grasp but quick to forget",
            "Sharp, clear, and precise",
            "Slow to learn but retains for long",
        ],
    },
    {
        "key": "speech",
        "text": "💬 How do you typically speak?",
        "options": [
            "Fast, talkative, jump from topic to topic",
            "Precise, direct, sharp, and articulate",
            "Slow, measured, melodious, sometimes silent",
        ],
    },
    {
        "key": "weather_preference",
        "text": "🌤️ What weather do you prefer?",
        "options": [
            "Warm and humid — dislike cold and wind",
            "Cool — dislike heat, sun, and hot weather",
            "Warm and dry — dislike cold, damp, and cloudy days",
        ],
    },
    {
        "key": "thirst",
        "text": "💧 How is your thirst?",
        "options": [
            "Variable — sometimes forget to drink water",
            "Excessive — always thirsty",
            "Low — rarely feel thirsty",
        ],
    },
    {
        "key": "sweat",
        "text": " How much do you sweat?",
        "options": [
            "Minimal sweating",
            "Profuse sweating, often with strong odour",
            "Moderate sweating",
        ],
    },
    {
        "key": "joints",
        "text": "🦴 How are your joints?",
        "options": [
            "Prominent, cracking, or popping joints",
            "Flexible and medium-sized",
            "Large, well-padded, stable joints",
        ],
    },
    {
        "key": "weight_tendency",
        "text": "⚖️ What is your weight tendency ?",
        "options": [
            "Hard to gain weight, lose it quickly",
            "Moderate — gain and lose with relative ease",
            "Easy to gain weight, very hard to lose it",
        ],
    },
]

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prakriti Determine",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .main { background: linear-gradient(135deg, #f5f0eb 0%, #e8f5e9 100%); }
    
    .chat-bubble-bot {
        background: rgba(91, 158, 110, 0.12);
        color: inherit;
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        margin: 8px 0 8px 0;
        max-width: 85%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        border-left: 4px solid #5B9E6E;
        font-size: 15px;
        line-height: 1.6;
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 8px 0 8px auto;
        max-width: 80%;
        text-align: right;
        box-shadow: 0 2px 8px rgba(102,126,234,0.35);
        font-size: 15px;
    }
    .chat-container { max-width: 700px; margin: 0 auto; }
    
    .prakriti-card {
        border-radius: 20px;
        padding: 28px;
        margin: 16px 0;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    .prakriti-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 8px 0;
    }
    .section-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.10);
    }
    .food-item {
        padding: 4px 0;
        font-size: 15px;
    }
    .progress-text {
        font-size: 13px;
        color: #666;
        text-align: center;
        margin-top: 4px;
    }
    .stButton > button {
        border-radius: 12px !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    .header-banner {
        background: linear-gradient(135deg, #2D5016 0%, #5B9E6E 50%, #A8D5B5 100%);
        border-radius: 20px;
        padding: 30px 24px;
        text-align: center;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(43,80,22,0.25);
    }
    .header-banner h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .header-banner p { font-size: 1rem; opacity: 0.9; margin: 8px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ─── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],          # list of {"role": "bot"|"user", "text": str}
        "q_index": 0,            # current question index
        "answers": [],           # collected answers (0/1/2 per question)
        "phase": "welcome",      # welcome | quiz | result
        "prediction": None,
        "probabilities": None,
        "show_diet": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def add_message(role, text):
    st.session_state.messages.append({"role": role, "text": text})

def reset_app():
    for k in ["messages", "q_index", "answers", "phase", "prediction", "probabilities", "show_diet"]:
        del st.session_state[k]
    init_state()

def predict_prakriti(answers):
    X = np.array(answers).reshape(1, -1)
    proba = MODEL.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    return PRAKRITI_LABELS[pred_idx], proba


# ─── Simple rule-based chatbot ────────────────────────────────────────────────
def chatbot_respond(question, prakriti, info, diet):
    q = question.lower()

    if any(w in q for w in ["what is", "explain", "tell me", "about", "meaning", "prakriti"]):
        return (
            f"**{prakriti} Prakriti** is governed by the {info['element']} elements. 🌿\n\n"
            f"{info['description']}\n\n"
            f"**Key traits:** {info['traits']}"
        )
    elif any(w in q for w in ["eat", "food", "diet", "meal", "nutrition"]):
        favour_str = "\n".join(diet["favor"][:5])
        return (
            f"For **{prakriti}** types, the key principle is: *{diet['principle']}*\n\n"
            f"**Top foods to favour:**\n{favour_str}\n\n"
            f"Would you like tips on what to avoid or meal timing?"
        )
    elif any(w in q for w in ["avoid", "don't", "bad", "harmful", "wrong"]):
        avoid_str = "\n".join(diet["avoid"])
        return (
            f"**{prakriti}** types should try to avoid:\n\n{avoid_str}\n\n"
            f"These can aggravate your dosha and lead to imbalance. 🙏"
        )
    elif any(w in q for w in ["tip", "advice", "suggestion", "meal", "time", "when"]):
        tips_str = "\n".join(diet["meal_tips"])
        return f"**Meal tips for {prakriti}:**\n\n{tips_str}"
    elif any(w in q for w in ["herb", "supplement", "plant", "medicine"]):
        return (
            f"Recommended herbs for **{prakriti}** Prakriti:\n\n"
            f"🌿 **{diet['herbs']}**\n\n"
            f"These herbs help balance your dominant dosha. Always consult an Ayurvedic practitioner before starting any supplement."
        )
    elif any(w in q for w in ["vata", "pitta", "kapha", "dosha", "three"]):
        return (
            "The three Doshas in Ayurveda are:\n\n"
            "🌬️ Vata — Air & Space — governs movement and communication \n\n"
            "🔥 Pitta — Fire & Water — governs transformation and metabolism\n\n"
            "🌊 Kapha— Earth & Water — governs structure and lubrication\n\n"
            f"Your dominant dosha is **{prakriti}** {info['emoji']}"
        )
    elif any(w in q for w in ["balance", "imbalance", "fix", "heal", "help"]):
        return (
            f"To keep your **{prakriti}** dosha in balance:\n\n"
            f"• Follow the diet recommendations above\n"
            f"• Maintain regular daily routines (*Dinacharya*)\n"
            f"• Get adequate sleep and avoid stress\n"
            f"• Use the herbs: {diet['herbs']}\n"
            f"• Consider consulting an Ayurvedic practitioner for a personalised plan 🙏"
        )
    elif any(w in q for w in ["exercise", "workout", "yoga", "activity"]):
        ex = {
            "Vata": "Gentle, grounding exercises — yoga, walking, tai chi, swimming. Avoid exhausting high-intensity workouts.",
            "Pitta": "Moderate, cooling exercises — swimming, cycling, hiking in cool weather. Avoid overheating and excessive competition.",
            "Kapha": "Vigorous, stimulating exercises — running, aerobics, weight training, HIIT. Kapha benefits most from intense activity!",
        }
        return f"Exercise for {prakriti} Prakriti:\n\n🏃 {ex[prakriti]}"
    elif any(w in q for w in ["sleep", "rest", "bedtime", "wake"]):
        sl = {
            "Vata": "Aim for 7–8 hours. Go to bed by 10 PM. Create a calming bedtime routine — warm milk, light reading.",
            "Pitta": "Aim for 7–8 hours. Avoid working late. Cool the bedroom and avoid stimulating content before bed.",
            "Kapha": "Limit to 7–8 hours maximum. Avoid oversleeping — it increases Kapha. Wake up by 6 AM ideally.",
        }
        return f"Sleep tips for {prakriti}:\n\n😴 {sl[prakriti]}"
    else:
        return (
            f"Great question! As a **{prakriti}** type, everything in Ayurveda is personalised to your constitution.\n\n"
            f"You can ask me about:\n"
            f"• 🥗 Diet and foods to eat or avoid\n"
            f"• 🌿 Herbs and supplements\n"
            f"• 🏃 Best exercises for your type\n"
            f"• 😴 Sleep recommendations\n"
            f"• ⚖️ Balancing your dosha\n\n"
            f"What would you like to know? 🙏"
        )


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🌿 Prakriti Determine</h1>
    <p>Discover your Ayurvedic body type through a guided conversation</p>
</div>
""", unsafe_allow_html=True)

# ─── Welcome phase ────────────────────────────────────────────────────────────
if st.session_state.phase == "welcome":
    if not st.session_state.messages:
        add_message("bot", (
            "Namaste! 🙏 Welcome to **Prakriti Determine**.\n\n"
            "In Ayurveda, *Prakriti* is your unique mind-body constitution — "
            "a blend of three life forces called **Doshas**: "
            "🌬️ **Vata** (Air & Space), 🔥 **Pitta** (Fire & Water), and 🌊 **Kapha** (Earth & Water).\n\n"
            "I'll ask you **15 simple questions** about your body, mind, and habits. "
            "Answer honestly based on your *natural tendencies*, not how you feel today.\n\n"
            "At the end, our AI model will predict your Prakriti and suggest a personalised diet! 🌱"
        ))

    # Display messages
    for msg in st.session_state.messages:
        cls = "chat-bubble-bot" if msg["role"] == "bot" else "chat-bubble-user"
        st.markdown(f'<div class="{cls}">{msg["text"]}</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start My Prakriti Journey", use_container_width=True):
            add_message("user", "Let's begin!")
            st.session_state.phase = "quiz"
            st.rerun()

# ─── Quiz phase ───────────────────────────────────────────────────────────────
elif st.session_state.phase == "quiz":
    # Show previous messages
    for msg in st.session_state.messages:
        cls = "chat-bubble-bot" if msg["role"] == "bot" else "chat-bubble-user"
        st.markdown(f'<div class="{cls}">{msg["text"]}</div>', unsafe_allow_html=True)

    q_idx = st.session_state.q_index
    total = len(QUESTIONS)

    if q_idx < total:
        q = QUESTIONS[q_idx]

        # Progress bar
        progress = q_idx / total
        st.progress(progress)
        st.markdown(f'<p class="progress-text">Question {q_idx + 1} of {total}</p>', unsafe_allow_html=True)

        # Show current question as bot bubble
        st.markdown(f'<div class="chat-bubble-bot">❓ {q["text"]}</div>', unsafe_allow_html=True)

        # Option buttons
        for i, option in enumerate(q["options"]):
            if st.button(option, key=f"opt_{q_idx}_{i}", use_container_width=True):
                # Save answer
                add_message("bot", f"❓ {q['text']}")
                add_message("user", option)
                st.session_state.answers.append(i)
                st.session_state.q_index += 1

                if st.session_state.q_index >= total:
                    # All answered — predict
                    prakriti, proba = predict_prakriti(st.session_state.answers)
                    st.session_state.prediction = prakriti
                    st.session_state.probabilities = proba
                    add_message("bot", (
                        f"✨ Thank you for completing all the questions! "
                        f"Our AI model has analysed your responses…\n\n"
                        f"🔮 Calculating your Prakriti…"
                    ))
                    st.session_state.phase = "result"

                st.rerun()

# ─── Result phase ─────────────────────────────────────────────────────────────
elif st.session_state.phase == "result":
    prakriti = st.session_state.prediction
    proba = st.session_state.probabilities
    info = PRAKRITI_INFO[prakriti]
    diet = DIET_SUGGESTIONS[prakriti]

    # ── Prakriti Result Card ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="prakriti-card" style="background: linear-gradient(135deg, {info['color']}22 0%, {info['color']}44 100%); border: 2px solid {info['color']};">
        <div style="font-size:4rem">{info['emoji']}</div>
        <div class="prakriti-title" style="color:{info['color']};">
            Your Prakriti is {prakriti}
        </div>
        <div style="opacity:0.75; font-size:0.95rem; margin:8px 0;">
            {info['element']} Constitution
        </div>
        <div style="opacity:0.8; font-size:0.9rem; margin-top:12px; font-style:italic;">
            "{info['description']}"
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dosha Percentages ─────────────────────────────────────────────────────
    st.markdown("### 📊 Your Dosha Breakdown")
    cols = st.columns(3)
    dosha_colors = {"Vata": "#8B7FD4", "Pitta": "#E07B54", "Kapha": "#5B9E6E"}
    dosha_emojis = {"Vata": "🌬️", "Pitta": "🔥", "Kapha": "🌊"}
    for i, (label, p) in enumerate(zip(PRAKRITI_LABELS, proba)):
        with cols[i]:
            pct = int(p * 100)
            c = dosha_colors[label]
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12);
                        border-radius:14px; padding:16px; text-align:center;
                        box-shadow:0 4px 12px rgba(0,0,0,0.12); border-top:4px solid {c}">
                <div style="font-size:1.8rem">{dosha_emojis[label]}</div>
                <div style="font-weight:600; color:{c}; font-size:1.1rem">{label}</div>
                <div style="font-size:2rem; font-weight:700;">{pct}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(p)

    # ── Key Traits ────────────────────────────────────────────────────────────
    with st.expander("🌟 Your Key Traits & Qualities", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**✨ Traits:** {info['traits']}")
        with col2:
            st.markdown(f"**⚡ Qualities:** {info['qualities']}")

    st.markdown("---")

    # ── Diet Suggestions ──────────────────────────────────────────────────────
    st.markdown(f"## 🥗 Diet Recommendations for {prakriti} Prakriti")
    st.markdown(f"""
    <div class="section-card" style="border-left:4px solid {info['color']};">
        <strong>🎯 Core Principle:</strong> {diet['principle']}
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### ✅ Foods to Favour")
        favour_html = "".join(f'<div class="food-item">{item}</div>' for item in diet["favor"])
        st.markdown(f'<div class="section-card">{favour_html}</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown("### ❌ Foods to Avoid")
        avoid_html = "".join(f'<div class="food-item">{item}</div>' for item in diet["avoid"])
        st.markdown(f'<div class="section-card">{avoid_html}</div>', unsafe_allow_html=True)

    st.markdown("### 💡 Meal Tips")
    tips_html = "".join(f'<div class="food-item">{tip}</div>' for tip in diet["meal_tips"])
    st.markdown(f'<div class="section-card">{tips_html}</div>', unsafe_allow_html=True)

    st.markdown("### 🌿 Recommended Herbs & Supplements")
    st.markdown(f"""
    <div class="section-card" style="border-left:4px solid #5B9E6E;">
        <strong>🌱 {diet['herbs']}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Chatbot Follow-up ─────────────────────────────────────────────────────
    st.markdown("### 💬 Ask the Chatbot")
    st.markdown(f"""
    <div class="chat-bubble-bot">
        Wonderful! 🌿 You have discovered your <strong>{prakriti}</strong> Prakriti!<br><br>
        Feel free to ask me anything about your Prakriti, your diet recommendations, 
        Ayurvedic lifestyle tips, or anything else you'd like to know. I'm here to help! 🙏
    </div>
    """, unsafe_allow_html=True)

    # Chat history for Q&A
    if "chat_qa" not in st.session_state:
        st.session_state.chat_qa = []

    for m in st.session_state.chat_qa:
        cls = "chat-bubble-bot" if m["role"] == "bot" else "chat-bubble-user"
        st.markdown(f'<div class="{cls}">{m["text"]}</div>', unsafe_allow_html=True)

    user_q = st.chat_input("Ask about your Prakriti, diet, lifestyle…")
    if user_q:
        st.session_state.chat_qa.append({"role": "user", "text": user_q})
        # Simple rule-based chatbot for Prakriti follow-up
        answer = chatbot_respond(user_q, prakriti, info, diet)
        st.session_state.chat_qa.append({"role": "bot", "text": answer})
        st.rerun()

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Retake the Quiz", use_container_width=True):
            reset_app()
            st.rerun()


