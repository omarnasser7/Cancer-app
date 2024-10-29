import gradio as gr
import random

# رسائل حب متنوعة
love_messages = [
    "🌹 الباشمهندسه مريم احسن باشمهندسه علي الكوكب 💖",
    " التيم بدون البشمهندسه مريم لا شئ❤️",
    " الباشمهندسه مريم وكفي 💕",
    "✨ اما ان تكون فاشل او تكون الباشمهندسه مريم 🌠",
    "🌹 الباشمهندسه مريم 💖"
]

# دالة اختيار رسالة حب عشوائية
def show_love():
    return random.choice(love_messages)

# تصميم واجهة Gradio
app = gr.Interface(fn=show_love, inputs=None, outputs="text", title="💖 حكمة اليوم 💖", 
                   description="اضغط الزر لتحصل على حكمة اليوم ✨")

# تشغيل التطبيق
app.launch()
