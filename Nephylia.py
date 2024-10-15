from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Charger le modèle et le tokenizer DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Salut! Pose-moi ta question.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text

    # Tokenizer l'entrée de l'utilisateur
    input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')

    # Générer la réponse avec le modèle DialoGPT
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    await update.message.reply_text(response)

if __name__ == '__main__':
    application = ApplicationBuilder().token("7775434320:AAFRLss3MCPcGnOcNn5oWO5_OWGdGacU9Us").build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    application.run_polling()
