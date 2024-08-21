import json
import os
import html
import re
import threading
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.core.text import LabelBase
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.button import ButtonBehavior
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.properties import BooleanProperty
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.clock import Clock
from google.cloud import translate_v2 as translate
from google.api_core.exceptions import BadRequest
import nltk 
import torch
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer, PegasusTokenizer, PegasusForConditionalGeneration

nltk.download('punkt')
nltk.download('stopwords')

# Initialize the summarization pipeline (This uses a model like T5 for paraphrasing)
def init_model():
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    return pipeline("summarization", model=model, tokenizer=tokenizer)

summarizer_pipeline = init_model()

paraphrase_pipeline = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

def apply_tone(text, tone):
    if tone == 'Formal':
        prompt = f"{text}"
    elif tone == 'Casual':
        prompt = f"{text}"
    else:
        return text
    
    paraphrased = paraphrase_pipeline(prompt, max_length=128, num_return_sequences=1, do_sample=False)[0]['generated_text']
    return paraphrased

def is_email(text):
    return bool(re.search(r'\b(?:@|\bDear\b|\bSincerely\b|\bBest\b|\bRegards\b|\bSubject\b)', text))

def extract_key_email_sentences(text):
    # Detect common phrases in emails
    action_phrases = ["please", "kindly", "urgent", "as soon as possible", "action required"]
    sentences = nltk.sent_tokenize(text)
    key_sentences = [sentence for sentence in sentences if any(phrase in sentence.lower() for phrase in action_phrases)]
    
    # Add the subject line if detected
    subject_line = re.search(r'(Subject:.*?)(\n|$)', text)
    if subject_line:
        key_sentences.insert(0, subject_line.group(1))

    return key_sentences

def score_sentences(text):
    sentences = nltk.sent_tokenize(text)
    # Use the summarizer model to score each sentence
    scores = []
    for sentence in sentences:
        score = summarizer_pipeline(sentence, max_new_tokens=10, min_new_tokens=5, do_sample=False)
        scores.append((sentence, len(score[0]['summary_text'])))  # Use length as a proxy for importance
    return scores

def summarize_text(text, n_sentences=None, custom_stopwords=None, detail_level=None, tone=None, is_email_summary=False):
    # Apply custom stopwords if provided
    if custom_stopwords:
        for stopword in custom_stopwords:
            text = text.replace(stopword, '')

    if is_email_summary and is_email(text):
        selected_sentences = extract_key_email_sentences(text)

    scored_sentences = score_sentences(text)
    sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
    
    # Adjust number of sentences based on detail level
    if detail_level == 'Brief':
        n_sentences = max(1, n_sentences // 2)
        selected_sentences = [sent for sent, score in sorted_sentences[:n_sentences]]
        selected_sentences = [summarizer_pipeline(sentence, max_new_tokens=25, min_new_tokens=10, do_sample=False)[0]['summary_text'] for sentence in selected_sentences]
    elif detail_level == 'Detailed':
        n_sentences = min(len(sorted_sentences), n_sentences * 2)
        selected_sentences = [sent for sent, score in sorted_sentences[:n_sentences]]
        # Expand the selected sentences
        expanded_sentences = [summarizer_pipeline(sentence, max_new_tokens=50, min_new_tokens=20, do_sample=False)[0]['summary_text'] for sentence in selected_sentences]
        selected_sentences = expanded_sentences
    else:
        selected_sentences = [sent for sent, score in sorted_sentences[:n_sentences]]
    
    if tone in ['Casual', 'Formal']:
        selected_sentences = [apply_tone(sentence, tone) for sentence in selected_sentences]

    summary = ' '.join(selected_sentences)
    return summary, selected_sentences


def refine_bullet_points(selected_sentences):
    bullet_points = selected_sentences
    return bullet_points

class CustomSpinnerOption(SpinnerOption):
    pass

class CustomSpinner(Spinner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropdown_cls = CustomDropDown

class CustomDropDown(DropDown):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_height = 5 * 44  # Set the max height for 5 items

class SettingsPopup(Popup):
    def __init__(self, app, **kwargs):
        super(SettingsPopup, self).__init__(**kwargs)
        self.app = app

    def on_stopwords_text(self, instance, value):
        self.app.set_stopwords(value)

    def on_length_slider_value(self, instance, value):
        self.app.summary_settings['length'] = int(value)
        self.ids.length_label.text = f'Number of Sentences: {str(int(value))}'
        
    def on_detail_dropdown_select(self, instance, value):
        self.app.summary_settings['detail'] = value

    def on_tone_dropdown_select(self, instance, value):
        self.app.summary_settings['tone'] = value

    def on_use_stopwords_toggle(self, instance, active):
        self.app.summary_settings['use_stopwords'] = active

    def on_save(self):
        self.dismiss()



class TextSummarizerApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.language_map = {
            'Auto Detect': 'auto',
            'English (en)': 'en', 'Chinese (zh)': 'zh', 'Hindi (hi)': 'hi', 'Spanish (es)': 'es',
            'French (fr)': 'fr', 'Arabic (ar)': 'ar', 'Bengali (bn)': 'bn', 'Russian (ru)': 'ru',
            'Portuguese (pt)': 'pt', 'Indonesian (id)': 'id', 'German (de)': 'de', 'Japanese (ja)': 'ja',
            'Telugu (te)': 'te', 'Marathi (mr)': 'mr', 'Turkish (tr)': 'tr', 'Tamil (ta)': 'ta',
            'Korean (ko)': 'ko', 'Vietnamese (vi)': 'vi', 'Urdu (ur)': 'ur', 'Italian (it)': 'it',
            'Persian (fa)': 'fa', 'Polish (pl)': 'pl', 'Ukrainian (uk)': 'uk', 'Malay (ms)': 'ms',
            'Thai (th)': 'th', 'Romanian (ro)': 'ro', 'Dutch (nl)': 'nl', 'Greek (el)': 'el', 'Swedish (sv)': 'sv'
        }
        self.reverse_language_map = {v: k for k, v in self.language_map.items()}  # Reverse mapping for displaying

        self.custom_stopwords = set()
        self.summary_settings = {
            'length': 1,         # Default number of sentences
            'detail': 'Normal',  # Options: 'brief', 'normal', 'detailed'
            'tone': 'Neutral',   # Options: 'formal', 'casual', 'neutral'
            'use_stopwords': True,
        }

    def build(self):
        self.tooltip = None
        with open('texts.json', 'r') as file:
            self.texts = json.load(file)['texts']
        return self.root
    
    def open_settings_popup(self):
        # Open the settings popup
        settings_popup = SettingsPopup(app=self)
        settings_popup.open()
    
    def set_stopwords(self, stopwords_text):
        self.custom_stopwords = set(stopwords_text.split(','))

    def detect_language(self, text):
        try:
            language = detect(text)
            print(f"Detected language: {language}")  # Debug statement
            return language
        except Exception as e:
            return None

    def translate_text(self, text, source_lang, target_lang):
        # Check if source and target languages are the same
        if source_lang == target_lang:
            # No translation needed, return the original text
            return text

        # Initialize the Google Cloud Translate client
        translate_client = translate.Client()

        try:
            # Perform the translation
            result = translate_client.translate(text, source_language=source_lang, target_language=target_lang)

            # Decode any HTML entities to their original characters
            translated_text = html.unescape(result['translatedText'])

            return translated_text

        except BadRequest as e:
            print(f"Translation error: {e}")
            # Handle the error or return the original text as a fallback
            return text
    
    def update_spinner(self, spinner_id, search_text):
        search_text = search_text.lower()
        filtered_options = [lang for lang in self.language_map.keys() if search_text in lang.lower()]
        spinner = self.root.ids[spinner_id]
        spinner.values = filtered_options
        if filtered_options:
            spinner.text = filtered_options[0]  # Set the first match as the current selection

    def get_lang_code(self, language_name):
        return self.language_map.get(language_name, 'en')  # Default to 'en' if not found
    
    def translate_text_button(self):
        source_lang_name = self.root.ids.source_lang_spinner.text
        target_lang_name = self.root.ids.target_lang_spinner.text

        # Handle "Auto" option for source language
        if source_lang_name == 'Auto Detect':
            source_lang = 'auto'
        else:
            source_lang = self.get_lang_code(source_lang_name)

        target_lang = self.get_lang_code(target_lang_name)

        input_text = self.root.ids.text_input.text.strip()

        # If source_lang is 'auto', detect the language
        if source_lang == 'auto':
            detected_language = self.detect_language(input_text)
            source_lang = detected_language

        # Show the loading popup
        self.show_loading(source_lang=source_lang, target_lang=target_lang)

        def perform_translation():
            translated_text = self.translate_text(input_text, source_lang, target_lang)
            
            # Update the input TextInput with the translated text on the main thread
            Clock.schedule_once(lambda dt: setattr(self.root.ids.text_input, 'text', translated_text), 0)
            Clock.schedule_once(lambda dt: self.hide_loading(), 0)

        # Run the translation in a separate thread
        threading.Thread(target=perform_translation).start()
    
    def summarize_text(self):
        self.show_loading()
        
        def perform_summarization():
            input_text = self.root.ids.text_input.text.strip()  # Use the updated translated text
            n_sentences = self.summary_settings['length']  # Use the value from settings

            # Perform the summarization with custom settings
            summary, summary_sentences = summarize_text(
                text=input_text, 
                n_sentences=n_sentences,
                custom_stopwords=self.custom_stopwords if self.summary_settings['use_stopwords'] else None,
                detail_level=self.summary_settings['detail'],
                tone=self.summary_settings['tone']
            )

            refined_bullet_points = refine_bullet_points(summary_sentences)
            best_line = refined_bullet_points[0] if refined_bullet_points else ""

            # Ensure both the summary and bullet points are correctly updated
            self.summary = summary
            self.summary_sentences = summary_sentences
            self.refined_bullet_points = refined_bullet_points
            self.best_line = best_line

            # Update title immediately after generating the summary
            title = f"Summary of: {self.current_title}" if hasattr(self, 'current_title') else "Summary:"
            Clock.schedule_once(lambda dt: self.update_summary_output(title), 0)
            Clock.schedule_once(lambda dt: setattr(self.root.ids.summary_output, 'text', summary), 0)

            # Ensure both summary and bullet points are accessible
            self.summary_generated = True

            if not self.summary_generated:
                Clock.schedule_once(lambda dt: setattr(self.root.ids.summarize_button, 'text', "Summarize Again"), 0)
                self.summary_generated = True

            Clock.schedule_once(lambda dt: self.hide_loading(), 0)

        threading.Thread(target=perform_summarization).start()

    def plot_summary_lengths(self):
        summary_lengths = self.summary_data['Summary'].apply(len)
        plt.figure(figsize=(8, 6))
        plt.hist(summary_lengths, bins=10, color='skyblue')
        plt.title('Distribution of Summary Lengths')
        plt.xlabel('Length of Summaries')
        plt.ylabel('Frequency')
        plt.show()
        
    def on_start(self):
        self.initialize_userdata()
        self.root.ids.spinner.values = [text['title'] for text in self.texts]
        self.summary_generated = False

    def save_userdata(self):
        with open('userdata.json', 'w') as file:
            json.dump(self.userdata, file, indent=4)

    def initialize_userdata(self):
        # Check if userdata.json exists, if not, create it with an empty structure
        if not os.path.exists('userdata.json'):
            with open('userdata.json', 'w') as file:
                json.dump({"summaries": {}}, file, indent=4)
        else:
            # If the file exists, check if it's empty
            with open('userdata.json', 'r+') as file:
                try:
                    # Try to load the data
                    self.userdata = json.load(file)
                except json.JSONDecodeError:
                    # If the file is empty or contains invalid JSON, initialize it
                    file.seek(0)
                    file.truncate()  # Clear the file content
                    self.userdata = {"summaries": {}}
                    json.dump(self.userdata, file, indent=4)

    def open_summary_selection_popup(self):
        # Create the popup
        popup = SummarySelectionPopup(app=self)
        summary_list = popup.ids.summary_list

        # Get the available summaries
        summaries = self.userdata.get('summaries', {})

        # Populate the RecycleView with summary names
        summary_list.data = [{'text': name, 'on_press': lambda x=name: self.on_summary_selected(x, popup)} for name in summaries.keys()]

        # Open the popup
        popup.open()

    def on_summary_selected(self, summary_name, popup):
        popup.dismiss()
        self.on_saved_summary_select(summary_name)
        
    def on_saved_summary_select(self, summary_name):
        if summary_name in self.userdata.get('summaries', {}):
            self.show_loading()

            def load_summary():
                self.summary = self.userdata['summaries'][summary_name]
                self.current_title = summary_name

                Clock.schedule_once(lambda dt: self.root.ids.summary_output.setter('text')(self.root.ids.summary_output, self.summary), 0)
                Clock.schedule_once(lambda dt: self.root.ids.summary_title.setter('text')(self.root.ids.summary_title, f"Summary of: {summary_name}"), 0)

                summary_sentences = nltk.sent_tokenize(self.summary)
                self.summary_sentences = summary_sentences
                self.refined_bullet_points = refine_bullet_points(summary_sentences)
                self.best_line = self.refined_bullet_points[0] if self.refined_bullet_points else ""

                Clock.schedule_once(lambda dt: setattr(self.root.ids.summary_output, 'focus', True), 0)

                self.hide_loading()

            threading.Thread(target=load_summary).start()


    def open_save_popup(self):
        self.save_popup = SavePopup()

        # Pre-fill the summary name if a summary is currently loaded
        if hasattr(self, 'current_title') and self.current_title:
            self.save_popup.ids.summary_name_input.text = self.current_title

        self.save_popup.open()

    def save_summary(self):
            self.summary = self.root.ids.summary_output.text
            summary_name = self.save_popup.ids.summary_name_input.text.strip()
            if not summary_name:
                self.save_popup.dismiss()
                return

            if summary_name in self.userdata['summaries']:
                self.confirmation_popup = ConfirmationPopup()
                self.confirmation_popup.open()
                self.summary_name_to_save = summary_name
            else:
                self._perform_save(summary_name)

            self.save_popup.dismiss()

    def confirm_overwrite(self):
        self._perform_save(self.summary_name_to_save)
        self.confirmation_popup.dismiss()

    def _perform_save(self, summary_name):
        try:
            with open('userdata.json', 'r+') as file:
                data = json.load(file)
                if "summaries" not in data:
                    data["summaries"] = {}

                data["summaries"][summary_name] = self.summary

                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate() 

                self.userdata = data 

                self.initialize_userdata()

        except FileNotFoundError:
            with open('userdata.json', 'w') as file:
                data = {"summaries": {summary_name: self.summary}}
                json.dump(data, file, indent=4)

                self.userdata = data 

                self.initialize_userdata()

    def on_text_select(self, text_title):
        selected_text = next((text for text in self.texts if text['title'] == text_title), None)
        if selected_text:
            original_language = self.detect_language(selected_text['content'])  
            if original_language not in ['sv', 'en']: 
                translated_text = self.translate_text(selected_text['content'], target_language='sv')
                self.root.ids.text_input.text = translated_text
                self.current_title = f"{selected_text['title']} (Translated from {original_language.upper()})"
            else:
                self.root.ids.text_input.text = selected_text['content']
                self.current_title = selected_text['title']
        else:
            self.root.ids.text_input.text = 'Text not found.'
            self.current_title = ''


    def show_loading(self, source_lang=None, target_lang=None):
        if source_lang and target_lang:
            loading_message = f"Translating from {self.reverse_language_map.get(source_lang, source_lang)[:-4]} to {self.reverse_language_map.get(target_lang, target_lang)[:-4]}..."
        else:
            loading_message = "Loading..."

        layout = BoxLayout(orientation='vertical')
        loading_image = Image(source='pictures/loading.gif', anim_delay=0.05)
        loading_label = Label(text=loading_message, size_hint_y=None, height=40)

        layout.add_widget(loading_image)
        layout.add_widget(loading_label)

        self.loading_popup = Popup(
            title="",
            content=layout,
            size_hint=(None, None),
            size=(300, 150),
            auto_dismiss=False,
            background='',
            background_color=(0, 0, 0, 0),
            separator_height=0
        )
        self.loading_popup.open()

    def hide_loading(self):
        if hasattr(self, 'loading_popup'):
            self.loading_popup.dismiss()
    
    def update_summary_output(self, text):
        if hasattr(self, 'current_title') and self.current_title:
            self.root.ids.summary_title.text = f"Summary of: {self.current_title}"
        
    def show_summary(self):
        if hasattr(self, 'summary') and self.summary_generated:
            self.root.ids.summary_output.text = self.summary
            if hasattr(self, 'current_title') and self.current_title:
                self.root.ids.summary_title.text = f"Summary of: {self.current_title}"
            else:
                self.root.ids.summary_title.text = "Summary:"
        else:
            self.root.ids.summary_output.text = "Please summarize text first."
            
    def show_bullet_points(self):
        if hasattr(self, 'refined_bullet_points') and self.refined_bullet_points:
            bullet_points_text = '\n'.join([f"• {point}\n" for point in self.refined_bullet_points])
            self.root.ids.summary_output.text = bullet_points_text
            self.root.ids.summary_title.text = "Bullet Points:"
        else:
            self.root.ids.summary_output.text = "Please summarize text first."

    def show_best_line(self):
        if hasattr(self, 'best_line') and self.best_line:
            self.root.ids.summary_output.text = f"• {self.best_line}"
            self.root.ids.summary_title.text = "Best Line:"
        else:
            self.root.ids.summary_output.text = "Please summarize text first."

    def update_slider_label(self, instance, value):
        self.root.ids.slider_label.text = f"Summarized Sentences: {int(value)}"

    def apply_formatting(self, format_type, tag=None):
        # Get the current selection (emulated here)
        text_input = self.root.ids.summary_output
        selected_text = text_input.text

        # Apply formatting
        if format_type == 'bold':
            formatted_text = f"**{selected_text}**"
        elif format_type == 'italic':
            formatted_text = f"*{selected_text}*"
        elif format_type == 'color':
            formatted_text = f"[{selected_text}]"  # Use a placeholder for color, since TextInput does not support direct color changes
        elif format_type == 'size':
            formatted_text = f"<{selected_text}>"  # Placeholder for size changes
        else:
            formatted_text = selected_text

        # Set the updated text with formatting applied
        text_input.text = formatted_text

class CustomTooltip(Label):
    pass

class ImageButton(ButtonBehavior, Image):
    tooltip_text = StringProperty('')

    def __init__(self, **kwargs):
        super(ImageButton, self).__init__(**kwargs)
        self.tooltip = None
        self.tooltip_event = None  # Event to manage delayed hiding of the tooltip
        self.bind(on_enter=self.on_hover)
        self.bind(on_leave=self.hide_tooltip)

    def on_hover(self, *args):
        if Window.mouse_pos:
            self.show_tooltip(self)

    def handle_touch(self, widget, touch):
        if widget.collide_point(*touch.pos):
            if not self.tooltip:
                self.show_tooltip(widget)
            else:
                self.update_tooltip_position(touch)
        else:
            self.hide_tooltip()

    def show_tooltip(self, widget, touch=None):
        if not self.tooltip:
            self.tooltip = CustomTooltip(
                text=widget.tooltip_text,
                size_hint=(None, None)
            )
            self.tooltip.texture_update()
            self.tooltip.size = (self.tooltip.texture_size[0], self.tooltip.texture_size[1])

            Window.add_widget(self.tooltip)
            if touch:
                self.update_tooltip_position(touch)
            else:
                # Use widget position if touch is not provided (e.g., for mouse hover)
                self.update_tooltip_position_with_fallback(widget.center_x + 10, widget.top + 10)

    def update_tooltip_position(self, touch):
        # Ensure tooltip stays within the window
        x = touch.x + 10
        y = touch.y + 10

        if x + self.tooltip.width > Window.width:
            x = touch.x - self.tooltip.width - 10  # Place tooltip to the left of the cursor
        if y + self.tooltip.height > Window.height:
            y = touch.y - self.tooltip.height - 10  # Place tooltip above the cursor

        self.tooltip.pos = (x, y)

    def update_tooltip_position_with_fallback(self, x, y):
        # Ensure tooltip stays within the window when using fallback positioning
        if x + self.tooltip.width > Window.width:
            x = x - self.tooltip.width - 20  # Place tooltip to the left of the widget
        if y + self.tooltip.height > Window.height:
            y = y - self.tooltip.height - 20  # Place tooltip above the widget

        self.tooltip.pos = (x, y)

    def hide_tooltip(self, *args):
        if self.tooltip:
            Window.remove_widget(self.tooltip)
            self.tooltip = None

class SummarySelectionPopup(Popup):
    app = ObjectProperty(None)

    def __init__(self, app, **kwargs):
        super(SummarySelectionPopup, self).__init__(**kwargs)
        self.app = app
        self.options_mode = False

    def delete_options(self):
        self.options_mode = not self.options_mode  # Toggle the options mode (show/hide checkboxes)
        if self.options_mode:
            self.ids.delete_button.text = 'Delete Marked Objects'
        else:
            self.ids.delete_button.text = 'Delete'

        # Update the data items in RecycleView
        for item in self.ids.summary_list.data:
            item['options_mode'] = self.options_mode
            item['marked'] = False  # Reset marked state

        # Refresh the RecycleView to apply the changes
        self.refresh_summary_list()

    def refresh_summary_list(self):
        summaries = self.app.userdata.get('summaries', {})
        self.ids.summary_list.data = [
            {
                'text': name,
                'marked': False,
                'options_mode': self.options_mode,
                'on_checkbox_active': self.on_checkbox_active
            }
            for name in summaries.keys()
        ]

    def on_checkbox_active(self, checkbox, value, item):
        # Find the item in the data list and update its 'marked' state
        for data_item in self.ids.summary_list.data:
            if data_item['text'] == item['text']:
                data_item['marked'] = value
                break

    def delete_marked_objects(self):
        if self.options_mode:
            marked_summaries = [item['text'] for item in self.ids.summary_list.data if item['marked']]
            for summary in marked_summaries:
                del self.app.userdata['summaries'][summary]
            self.app.save_userdata()
            self.refresh_summary_list()
            self.options_mode = False  # Exit options mode after deleting



class SummaryItem(BoxLayout):
    text = StringProperty()
    marked = BooleanProperty(False)
    options_mode = BooleanProperty(False)
    on_checkbox_active = None

    def __init__(self, **kwargs):
        super(SummaryItem, self).__init__(**kwargs)

    def on_checkbox_active(self, checkbox, value):
        if self.on_checkbox_active:
            self.on_checkbox_active(checkbox, value, self)

class SavePopup(Popup):
    pass

class ConfirmationPopup(Popup):
    pass

if __name__ == '__main__':
    TextSummarizerApp().run()
