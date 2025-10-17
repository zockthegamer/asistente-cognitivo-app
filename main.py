from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.core.audio import SoundLoader
import cv2
import face_recognition
import speech_recognition as sr
import pyttsx3
import firebase_admin
from firebase_admin import credentials, firestore, storage
import pickle
import numpy as np
import threading
import os
import time
import shutil
from datetime import datetime
import locale
import sys
from PIL import Image as PILImage, ImageDraw, ImageFont

try:
    if sys.platform.startswith('win'):
        locale.setlocale(locale.LC_TIME, 'Spanish')
    else:
        locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
    print("Locale para fechas configurado exitosamente en español.")
except locale.Error:
    print("Advertencia: No se pudo configurar el locale a español. La fecha podría aparecer en inglés.")

STORAGE_BUCKET_NAME = 'asistente-cognitivo.firebasestorage.app'
cred = credentials.Certificate("credenciales.json")
firebase_admin.initialize_app(cred, {'storageBucket': STORAGE_BUCKET_NAME})
db = firestore.client()
bucket = storage.bucket()
print("Conexión con Firebase (Firestore y Storage) inicializada.")

engine = pyttsx3.init()

class LoginScreen(Screen):
    pass

class ChoiceScreen(Screen):
    pass

class UserManagementScreen(Screen):
    def on_enter(self):
        self.populate_user_list()

    def populate_user_list(self):
        self.ids.user_list_layout.clear_widgets()
        users_ref = db.collection('usuarios').stream()
        for user in users_ref:
            user_id = user.id
            user_data = user.to_dict()
            user_name = user_data.get('nombre', user_id)
            user_layout = BoxLayout(size_hint_y=None, height=40)
            name_label = Label(text=user_name)
            delete_button = Button(text="Eliminar", size_hint_x=0.3, background_color=(1, 0.3, 0.3, 1))
            delete_button.bind(on_press=lambda instance, u_id=user_id, u_name=user_name: self.open_confirm_popup(u_id, u_name))
            user_layout.add_widget(name_label)
            user_layout.add_widget(delete_button)
            self.ids.user_list_layout.add_widget(user_layout)

    def open_confirm_popup(self, user_id, user_name):
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        popup_label = Label(text=f"¿Está seguro de que desea eliminar a\n'{user_name}'?\nEsta acción es irreversible.")
        buttons_layout = BoxLayout(size_hint_y=None, height=50)
        confirm_btn = Button(text="Sí, Eliminar", background_color=(0.8, 0.2, 0.2, 1))
        cancel_btn = Button(text="Cancelar")
        buttons_layout.add_widget(confirm_btn)
        buttons_layout.add_widget(cancel_btn)
        content.add_widget(popup_label)
        content.add_widget(buttons_layout)
        popup = Popup(title="Confirmar Eliminación", content=content, size_hint=(0.8, 0.4))
        confirm_btn.bind(on_press=lambda instance: (self.delete_user(user_id), popup.dismiss()))
        cancel_btn.bind(on_press=popup.dismiss)
        popup.open()

    def delete_user(self, user_id):
        threading.Thread(target=self._delete_in_background, args=(user_id,)).start()

    def _delete_in_background(self, user_id):
        print(f"Iniciando eliminación de usuario: {user_id}")
        try:
            db.collection('usuarios').document(user_id).delete()
            print(f"Usuario {user_id} eliminado de Firebase.")
        except Exception as e:
            print(f"Error al eliminar de Firebase: {e}")
        face_folder = os.path.join('dataset', user_id)
        if os.path.exists(face_folder):
            try: shutil.rmtree(face_folder)
            except Exception as e: print(f"Error al eliminar carpeta facial: {e}")
        voice_folder = os.path.join('voice_dataset', user_id)
        if os.path.exists(voice_folder):
            try: shutil.rmtree(voice_folder)
            except Exception as e: print(f"Error al eliminar carpeta de voz: {e}")
        Clock.schedule_once(lambda dt: self.populate_user_list())

class FacialLoginScreen(Screen):
    def on_enter(self, *args):
        self.known_face_encodings = []
        self.known_face_names = []
        try:
            users_ref = db.collection('usuarios').stream()
            for user in users_ref:
                user_data = user.to_dict()
                if 'face_encoding' in user_data:
                    self.known_face_names.append(user_data.get('nombre', 'Desconocido'))
                    self.known_face_encodings.append(pickle.loads(user_data['face_encoding']))
            print(f"Cargados {len(self.known_face_names)} usuarios para reconocimiento.")
        except Exception as e:
            print(f"Error al cargar usuarios de Firebase: {e}")
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            self.ids.feedback_label.text = "Buscando rostro..."
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    self.login_success(name)
                    return
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.camera_feed.texture = image_texture

    def login_success(self, user_name):
        Clock.unschedule(self.update)
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()
        user_id = user_name.lower()
        App.get_running_app().current_user = user_id
        try:
            doc_ref = db.collection('usuarios').document(user_id)
            user_doc = doc_ref.get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                role = user_data.get('rol', 'paciente')
                if role == 'cuidador':
                    self.manager.current = 'caregiver_dashboard'
                else:
                    self.manager.get_screen('main_menu').ids.welcome_label.text = f"Hola, {user_name}"
                    self.manager.current = 'main_menu'
            else:
                print(f"Advertencia: No se encontró el documento para el usuario {user_id}")
                self.manager.get_screen('main_menu').ids.welcome_label.text = f"Hola, {user_name}"
                self.manager.current = 'main_menu'
        except Exception as e:
            print(f"Error al obtener el rol del usuario: {e}")
            self.manager.get_screen('main_menu').ids.welcome_label.text = f"Hola, {user_name}"
            self.manager.current = 'main_menu'

    def on_leave(self, *args):
        Clock.unschedule(self.update, all=True)
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()

class FacialRegistrationScreen(Screen):
    def on_enter(self, *args):
        if not os.path.exists('dataset'): os.makedirs('dataset')
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        self.ids.action_button.disabled = False
        self.ids.feedback_label.text = 'Presione "Iniciar" para comenzar'
        self.frame = None

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            self.frame = frame
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.camera_feed.texture = image_texture

    def start_intelligent_capture(self):
        self.ids.action_button.disabled = True
        threading.Thread(target=self._registration_flow).start()

    def _registration_flow(self):
        nombre_usuario = ""
        nombre_confirmado = False
        while not nombre_confirmado:
            try:
                prompt_text = "Por favor, diga solo su primer nombre para el registro"
                self.update_label_safe(prompt_text)
                engine.say(prompt_text); engine.runAndWait()
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    self.update_label_safe("Adelante, estoy escuchando...")
                    engine.say("Adelante, estoy escuchando."); engine.runAndWait()
                    audio = r.listen(source, timeout=10, phrase_time_limit=10)
                    self.update_label_safe("Procesando...")
                    nombre_reconocido = r.recognize_google(audio, language='es-ES').capitalize()
                    confirm_prompt = f"He escuchado {nombre_reconocido}. ¿Es correcto? Diga sí o no."
                    self.update_label_safe(confirm_prompt)
                    engine.say(confirm_prompt); engine.runAndWait()
                    audio_confirm = r.listen(source, timeout=10, phrase_time_limit=5)
                    respuesta = r.recognize_google(audio_confirm, language='es-ES').lower()
                    if 's' in respuesta:
                        nombre_usuario = nombre_reconocido.lower().split()[0]
                        nombre_confirmado = True
                    else:
                        self.update_label_safe("Entendido. Intentémoslo de nuevo.")
                        time.sleep(2)
            except Exception as e:
                self.update_label_safe("No pude entender el audio. Vamos a intentarlo de nuevo.")
                print(f"Error de voz en confirmación: {e}")
                time.sleep(2)
        self._auto_capture_loop(nombre_usuario)

    def _auto_capture_loop(self, nombre_usuario):
        user_folder = os.path.join('dataset', nombre_usuario)
        if not os.path.exists(user_folder): os.makedirs(user_folder)
        photo_count = 0
        total_photos = 10
        master_encoding_saved = False
        while photo_count < total_photos:
            if self.frame is None:
                time.sleep(0.1)
                continue
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                img_name = os.path.join(user_folder, f"{photo_count + 1}.jpg")
                cv2.imwrite(img_name, self.frame)
                photo_count += 1
                self.update_label_safe(f"Foto {photo_count}/{total_photos} capturada.")
                if not master_encoding_saved:
                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    pickled_encoding = pickle.dumps(face_encoding)
                    doc_ref = db.collection('usuarios').document(nombre_usuario)
                    doc_ref.set({'nombre': nombre_usuario.capitalize(), 'face_encoding': pickled_encoding})
                    master_encoding_saved = True
                time.sleep(0.4)
            else:
                self.update_label_safe(f"Buscando rostro... ({photo_count}/{total_photos})")
                time.sleep(0.1)
        self.update_label_safe(f"¡Registro facial de {nombre_usuario.capitalize()} completado!")
        engine.say(f"Registro facial completado."); engine.runAndWait()
        App.get_running_app().registered_user = nombre_usuario
        Clock.schedule_once(self.go_to_role_selection, 2)

    def update_label_safe(self, text):
        Clock.schedule_once(lambda dt: self.update_label(text))
    def update_label(self, text):
        self.ids.feedback_label.text = text
    def go_to_role_selection(self, dt):
        self.manager.current = 'role_selection'
    def on_leave(self, *args):
        Clock.unschedule(self.update, all=True)
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()

class RoleSelectionScreen(Screen):
    def select_role(self, role):
        user_id = App.get_running_app().registered_user
        if not user_id:
            print("Error: No se encontró el usuario recién registrado.")
            self.manager.current = 'login'
            return
        try:
            doc_ref = db.collection('usuarios').document(user_id)
            doc_ref.update({'rol': role})
            print(f"Rol '{role}' asignado al usuario '{user_id}'.")
            App.get_running_app().registered_user = None
            self.manager.current = 'login'
        except Exception as e:
            print(f"Error al actualizar el rol en Firebase: {e}")
            self.manager.current = 'login'

class VoiceRegistrationScreen(Screen):
    pass

class MainMenuScreen(Screen):
    temp_audio_path = "temp_reminder.wav"

    def open_add_reminder_popup(self):
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        content.add_widget(Label(text='Título del recordatorio:'))
        self.title_input = TextInput(hint_text='Ej: Cita con el doctor')
        content.add_widget(self.title_input)
        self.status_label = Label(text='Presione "Grabar Voz" para dictar el contenido.')
        content.add_widget(self.status_label)
        buttons_layout = BoxLayout(size_hint_y=None, height=50)
        record_btn = Button(text="Grabar Voz")
        save_btn = Button(text="Guardar Recordatorio", disabled=True)
        buttons_layout.add_widget(record_btn)
        buttons_layout.add_widget(save_btn)
        content.add_widget(buttons_layout)
        popup = Popup(title="Agregar Nuevo Recordatorio", content=content, size_hint=(0.9, 0.6))
        record_btn.bind(on_press=lambda instance: threading.Thread(target=self.record_reminder_audio, args=(save_btn,)).start())
        save_btn.bind(on_press=lambda instance: self.save_reminder_and_close(popup))
        popup.open()

    def record_reminder_audio(self, save_btn):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                engine.say("Adelante, dicte el contenido del recordatorio."); engine.runAndWait()
                audio_data = r.listen(source, timeout=10, phrase_time_limit=15)
                with open(self.temp_audio_path, "wb") as f:
                    f.write(audio_data.get_wav_data())
                Clock.schedule_once(lambda dt: self.update_record_status("¡Grabación completada!", save_btn, is_error=False))
        except Exception as e:
            print(f"Error al grabar audio: {e}")
            Clock.schedule_once(lambda dt: self.update_record_status("Error al grabar. Intente de nuevo.", save_btn, is_error=True))

    def update_record_status(self, text, save_btn, is_error):
        self.status_label.text = text
        save_btn.disabled = is_error

    def save_reminder_and_close(self, popup):
        user_id = App.get_running_app().current_user
        reminder_title = self.title_input.text
        if user_id and reminder_title and os.path.exists(self.temp_audio_path):
            popup.content.children[0].children[1].disabled = True
            threading.Thread(target=self._upload_and_save, args=(user_id, reminder_title, popup)).start()

    def _upload_and_save(self, user_id, reminder_title, popup):
        destination_blob_name = f"reminders/{user_id}/{int(time.time())}.wav"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(self.temp_audio_path)
        doc_ref = db.collection('usuarios').document(user_id).collection('recordatorios').document()
        doc_ref.set({'titulo': reminder_title, 'audio_path': destination_blob_name, 'fecha_creacion': datetime.now()})
        os.remove(self.temp_audio_path)
        Clock.schedule_once(lambda dt: popup.dismiss())
        engine.say("Recordatorio guardado."); engine.runAndWait()

    def logout(self):
        App.get_running_app().current_user = None
        self.manager.current = 'login'

class CaregiverDashboardScreen(Screen):
    def on_enter(self, *args):
        user_name = App.get_running_app().current_user
        if user_name:
            self.ids.welcome_label.text = f"Hola, Cuidador {user_name.capitalize()}"
        self.load_patients()

    def load_patients(self):
        caregiver_id = App.get_running_app().current_user
        if not caregiver_id: return
        self.ids.patient_list_grid.clear_widgets()
        patients_ref = db.collection('usuarios').where('cuidador_id', '==', caregiver_id).stream()
        for patient in patients_ref:
            patient_id = patient.id
            patient_name = patient.to_dict().get('nombre', patient_id)
            btn = Button(
                text=patient_name,
                font_size='20sp',
                size_hint_y=None,
                height=50
            )
            btn.bind(on_press=lambda instance, p_id=patient_id, p_name=patient_name: self.select_patient(p_id, p_name))
            self.ids.patient_list_grid.add_widget(btn)

    def select_patient(self, patient_id, patient_name):
        App.get_running_app().selected_patient_id = patient_id
        self.manager.get_screen('patient_management').ids.patient_name_label.text = f"Recordatorios de {patient_name}"
        self.manager.current = 'patient_management'

    def logout(self):
        App.get_running_app().current_user = None
        App.get_running_app().selected_patient_id = None
        self.manager.current = 'login'

class AddPatientScreen(Screen):
    def search_patients(self):
        search_text = self.ids.search_input.text.strip().capitalize()
        caregiver_id = App.get_running_app().current_user
        self.ids.search_results_grid.clear_widgets()
        if not search_text:
            self.ids.status_label.text = "Por favor, ingrese un nombre para buscar."
            return
        self.ids.status_label.text = f"Buscando pacientes con el nombre '{search_text}'..."
        patients_ref = db.collection('usuarios').where('rol', '==', 'paciente').where('nombre', '==', search_text).stream()
        results_found = False
        for patient in patients_ref:
            results_found = True
            patient_id = patient.id
            patient_data = patient.to_dict()
            if 'cuidador_id' in patient_data and patient_data['cuidador_id']:
                if patient_data['cuidador_id'] == caregiver_id:
                    status_text = "Ya vinculado a usted"
                    btn_disabled = True
                else:
                    status_text = "Vinculado a otro cuidador"
                    btn_disabled = True
            else:
                status_text = "Disponible para vincular"
                btn_disabled = False
            item_layout = BoxLayout(size_hint_y=None, height=50)
            name_label = Label(text=f"{patient_data.get('nombre', patient_id)} ({status_text})")
            link_button = Button(text='Vincular', size_hint_x=0.3, disabled=btn_disabled)
            link_button.bind(on_press=lambda instance, p_id=patient_id, p_name=patient_data.get('nombre'): self.link_patient(p_id, p_name))
            item_layout.add_widget(name_label)
            item_layout.add_widget(link_button)
            self.ids.search_results_grid.add_widget(item_layout)
        if not results_found:
            self.ids.status_label.text = "No se encontraron pacientes con ese nombre."

    def link_patient(self, patient_id, patient_name):
        caregiver_id = App.get_running_app().current_user
        if not caregiver_id: return
        try:
            patient_ref = db.collection('usuarios').document(patient_id)
            patient_ref.update({'cuidador_id': caregiver_id})
            popup_content = BoxLayout(orientation='vertical', padding=20)
            popup_content.add_widget(Label(text=f"¡Éxito!\nAhora está vinculado a {patient_name}."))
            ok_button = Button(text="Aceptar", size_hint_y=None, height=50)
            popup_content.add_widget(ok_button)
            popup = Popup(title="Vinculación Completa", content=popup_content, size_hint=(0.8, 0.4))
            ok_button.bind(on_press=popup.dismiss)
            popup.open()
            self.ids.search_input.text = ""
            self.ids.search_results_grid.clear_widgets()
            self.manager.get_screen('caregiver_dashboard').load_patients()
        except Exception as e:
            print(f"Error al vincular paciente: {e}")
            self.ids.status_label.text = "Ocurrió un error al intentar vincular."

class RemindersListScreen(Screen):
    def on_enter(self):
        self.populate_reminders()

    def populate_reminders(self):
        self.ids.reminders_grid.clear_widgets()
        user_id = App.get_running_app().current_user
        if not user_id: return
        reminders_ref = db.collection('usuarios').document(user_id).collection('recordatorios').order_by('fecha_creacion').stream()
        for reminder in reminders_ref:
            reminder_id = reminder.id
            data = reminder.to_dict()
            item_layout = BoxLayout(size_hint_y=None, height=60, spacing=10)
            title_label = Label(text=data.get('titulo', 'Sin Título'), font_size='18sp', halign='left')
            item_layout.add_widget(title_label)
            listen_btn = Button(text="🔊", font_name='NotoEmoji-Regular.ttf', size_hint_x=None, width=50)
            edit_btn = Button(text="✏️", font_name='NotoEmoji-Regular.ttf', size_hint_x=None, width=50)
            delete_btn = Button(text="🗑️", font_name='NotoEmoji-Regular.ttf', size_hint_x=None, width=50, background_color=(1,0.5,0.5,1))
            listen_btn.bind(on_press=lambda instance, path=data.get('audio_path', ''): self.play_audio_reminder(path))
            edit_btn.bind(on_press=lambda instance, r_id=reminder_id, d=data: self.open_edit_popup(r_id, d))
            delete_btn.bind(on_press=lambda instance, r_id=reminder_id, title=data.get('titulo', ''), path=data.get('audio_path', ''): self.open_delete_popup(r_id, title, path))
            item_layout.add_widget(listen_btn)
            item_layout.add_widget(edit_btn)
            item_layout.add_widget(delete_btn)
            self.ids.reminders_grid.add_widget(item_layout)

    def play_audio_reminder(self, audio_path):
        if audio_path:
            threading.Thread(target=self._download_and_play, args=(audio_path, App.get_running_app().speech_lock)).start()

    def _download_and_play(self, audio_path, lock):
        with lock:
            try:
                blob = bucket.blob(audio_path)
                temp_dir = "temp_audio"
                if not os.path.exists(temp_dir): os.makedirs(temp_dir)
                destination_file_name = os.path.join(temp_dir, f"{int(time.time())}_{os.path.basename(audio_path)}")
                blob.download_to_filename(destination_file_name)
                sound = SoundLoader.load(destination_file_name)
                if sound:
                    sound.volume = 1.5
                    sound.play()
                    while sound.state == 'play':
                        time.sleep(0.1)
                os.remove(destination_file_name)
            except Exception as e:
                print(f"Error al reproducir el audio: {e}")
                engine.say("No se pudo reproducir el recordatorio."); engine.runAndWait()

    def open_delete_popup(self, reminder_id, reminder_title, audio_path):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(text=f"¿Seguro que desea eliminar:\n'{reminder_title}'?"))
        buttons = BoxLayout(size_hint_y=None, height=50)
        yes_btn = Button(text="Sí, Eliminar", background_color=(0.8,0.2,0.2,1))
        no_btn = Button(text="No")
        buttons.add_widget(yes_btn)
        buttons.add_widget(no_btn)
        content.add_widget(buttons)
        popup = Popup(title="Confirmar", content=content, size_hint=(0.8, 0.4))
        yes_btn.bind(on_press=lambda instance: (self._confirm_delete(reminder_id, audio_path), popup.dismiss()))
        no_btn.bind(on_press=popup.dismiss)
        popup.open()

    def _confirm_delete(self, reminder_id, audio_path):
        threading.Thread(target=self._delete_in_background, args=(reminder_id, audio_path)).start()

    def _delete_in_background(self, reminder_id, audio_path):
        user_id = App.get_running_app().current_user
        if user_id:
            try:
                db.collection('usuarios').document(user_id).collection('recordatorios').document(reminder_id).delete()
                if audio_path:
                    blob = bucket.blob(audio_path)
                    blob.delete()
                Clock.schedule_once(lambda dt: self.populate_reminders())
            except Exception as e:
                print(f"Error al eliminar: {e}")

    def open_edit_popup(self, reminder_id, data):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        title_input = TextInput(text=data.get('titulo', ''))
        content.add_widget(Label(text='Nuevo Título:'))
        content.add_widget(title_input)
        save_btn = Button(text="Guardar Cambios", size_hint_y=None, height=50)
        content.add_widget(save_btn)
        popup = Popup(title="Editar Título", content=content, size_hint=(0.9, 0.4))
        save_btn.bind(on_press=lambda instance: (self._save_edit(reminder_id, title_input.text), popup.dismiss()))
        popup.open()

    def _save_edit(self, reminder_id, new_title):
        threading.Thread(target=self._save_in_background, args=(reminder_id, new_title)).start()

    def _save_in_background(self, reminder_id, new_title):
        user_id = App.get_running_app().current_user
        if user_id and new_title:
            doc_ref = db.collection('usuarios').document(user_id).collection('recordatorios').document(reminder_id)
            doc_ref.update({'titulo': new_title})
            Clock.schedule_once(lambda dt: self.populate_reminders())

class OrientationScreen(Screen):
    def on_enter(self):
        self.update_time()
        Clock.schedule_interval(self.update_time, 1)
        threading.Thread(target=self.speak_date_and_time).start()

    def update_time(self, *args):
        now = datetime.now()
        self.ids.date_label.text = now.strftime('%A, %d de %B de %Y').capitalize()
        self.ids.time_label.text = now.strftime('%I:%M:%S %p')

    def speak_date_and_time(self):
        with App.get_running_app().speech_lock:
            now = datetime.now()
            date_text = now.strftime('Hoy es %A, %d de %B de %Y')
            time_text = now.strftime('y son las %I:%M %p')
            local_engine = pyttsx3.init()
            local_engine.say(date_text)
            local_engine.say(time_text)
            local_engine.runAndWait()

    def on_leave(self):
        Clock.unschedule(self.update_time)

class PatientManagementScreen(Screen):
    temp_audio_path = "temp_caregiver_reminder.wav"

    def on_enter(self):
        self.populate_reminders()

    def populate_reminders(self):
        self.ids.reminders_grid.clear_widgets()
        patient_id = App.get_running_app().selected_patient_id
        if not patient_id: return
        reminders_ref = db.collection('usuarios').document(patient_id).collection('recordatorios').order_by('fecha_creacion').stream()
        for reminder in reminders_ref:
            reminder_id = reminder.id
            data = reminder.to_dict()
            item_layout = BoxLayout(size_hint_y=None, height=60, spacing=10)
            title_label = Label(text=data.get('titulo', 'Sin Título'), font_size='18sp', halign='left')
            item_layout.add_widget(title_label)
            listen_btn = Button(text="🔊", font_name='NotoEmoji-Regular.ttf', size_hint_x=None, width=50)
            delete_btn = Button(text="🗑️", font_name='NotoEmoji-Regular.ttf', size_hint_x=None, width=50, background_color=(1,0.5,0.5,1))
            listen_btn.bind(on_press=lambda instance, path=data.get('audio_path', ''): self.play_audio_reminder(path))
            delete_btn.bind(on_press=lambda instance, r_id=reminder_id, title=data.get('titulo', ''), path=data.get('audio_path', ''): self.open_delete_popup(r_id, title, path))
            item_layout.add_widget(listen_btn)
            item_layout.add_widget(delete_btn)
            self.ids.reminders_grid.add_widget(item_layout)

    def open_add_reminder_popup(self):
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        content.add_widget(Label(text='Título del recordatorio:'))
        self.title_input = TextInput(hint_text='Ej: Tomar pastilla azul')
        content.add_widget(self.title_input)
        self.status_label = Label(text='Presione "Grabar Voz" para dictar el contenido.')
        content.add_widget(self.status_label)
        buttons_layout = BoxLayout(size_hint_y=None, height=50)
        record_btn = Button(text="Grabar Voz")
        save_btn = Button(text="Guardar Recordatorio", disabled=True)
        buttons_layout.add_widget(record_btn)
        buttons_layout.add_widget(save_btn)
        content.add_widget(buttons_layout)
        popup = Popup(title="Nuevo Recordatorio para el Paciente", content=content, size_hint=(0.9, 0.6))
        record_btn.bind(on_press=lambda instance: threading.Thread(target=self.record_reminder_audio, args=(save_btn,)).start())
        save_btn.bind(on_press=lambda instance: self.save_reminder_and_close(popup))
        popup.open()

    def record_reminder_audio(self, save_btn):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                engine.say("Adelante, dicte el contenido del recordatorio."); engine.runAndWait()
                audio_data = r.listen(source, timeout=10, phrase_time_limit=15)
                with open(self.temp_audio_path, "wb") as f: f.write(audio_data.get_wav_data())
                Clock.schedule_once(lambda dt: self.update_record_status("¡Grabación completada!", save_btn, False))
        except Exception as e:
            print(f"Error al grabar audio: {e}")
            Clock.schedule_once(lambda dt: self.update_record_status("Error al grabar.", save_btn, True))

    def update_record_status(self, text, save_btn, is_error):
        self.status_label.text = text
        save_btn.disabled = is_error

    def save_reminder_and_close(self, popup):
        patient_id = App.get_running_app().selected_patient_id
        reminder_title = self.title_input.text
        if patient_id and reminder_title and os.path.exists(self.temp_audio_path):
            popup.content.children[0].children[1].disabled = True
            threading.Thread(target=self._upload_and_save, args=(patient_id, reminder_title, popup)).start()

    def _upload_and_save(self, patient_id, reminder_title, popup):
        destination_blob_name = f"reminders/{patient_id}/{int(time.time())}.wav"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(self.temp_audio_path)
        doc_ref = db.collection('usuarios').document(patient_id).collection('recordatorios').document()
        doc_ref.set({'titulo': reminder_title, 'audio_path': destination_blob_name, 'fecha_creacion': datetime.now()})
        os.remove(self.temp_audio_path)
        Clock.schedule_once(lambda dt: popup.dismiss())
        Clock.schedule_once(lambda dt: self.populate_reminders())
        engine.say("Recordatorio guardado para el paciente."); engine.runAndWait()

    def play_audio_reminder(self, audio_path):
        if audio_path:
            threading.Thread(target=self._download_and_play, args=(audio_path, App.get_running_app().speech_lock)).start()

    def _download_and_play(self, audio_path, lock):
        with lock:
            try:
                blob = bucket.blob(audio_path)
                temp_dir = "temp_audio"; os.makedirs(temp_dir, exist_ok=True)
                destination_file_name = os.path.join(temp_dir, f"{int(time.time())}.wav")
                blob.download_to_filename(destination_file_name)
                sound = SoundLoader.load(destination_file_name)
                if sound:
                    sound.play()
                    while sound.state == 'play': time.sleep(0.1)
                os.remove(destination_file_name)
            except Exception as e:
                print(f"Error al reproducir el audio: {e}")

    def open_delete_popup(self, reminder_id, reminder_title, audio_path):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(text=f"¿Eliminar el recordatorio:\n'{reminder_title}'?"))
        buttons = BoxLayout(size_hint_y=None, height=50)
        yes_btn = Button(text="Sí, Eliminar", background_color=(0.8,0.2,0.2,1))
        no_btn = Button(text="No")
        buttons.add_widget(yes_btn); buttons.add_widget(no_btn)
        content.add_widget(buttons)
        popup = Popup(title="Confirmar", content=content, size_hint=(0.8, 0.4))
        yes_btn.bind(on_press=lambda instance: (self._confirm_delete(reminder_id, audio_path), popup.dismiss()))
        no_btn.bind(on_press=popup.dismiss)
        popup.open()

    def _confirm_delete(self, reminder_id, audio_path):
        threading.Thread(target=self._delete_in_background, args=(reminder_id, audio_path)).start()

    def _delete_in_background(self, reminder_id, audio_path):
        patient_id = App.get_running_app().selected_patient_id
        if patient_id:
            try:
                db.collection('usuarios').document(patient_id).collection('recordatorios').document(reminder_id).delete()
                if audio_path: bucket.blob(audio_path).delete()
                Clock.schedule_once(lambda dt: self.populate_reminders())
            except Exception as e:
                print(f"Error al eliminar: {e}")

class RecognitionScreen(Screen):
    active_trackers = []
    last_spoken_times = {}
    cooldown_period = 30
    frame_counter = 0

    def on_enter(self, *args):
        self.frame_counter = 0
        self.active_trackers = []
        self.last_spoken_times = {}
        self.known_face_encodings = []
        self.known_face_names_db = []
        try:
            users_ref = db.collection('usuarios').stream()
            for user in users_ref:
                user_data = user.to_dict()
                if 'face_encoding' in user_data:
                    self.known_face_names_db.append(user_data.get('nombre', 'Desconocido'))
                    self.known_face_encodings.append(pickle.loads(user_data['face_encoding']))
            print(f"Cargados {len(self.known_face_names_db)} usuarios para reconocimiento.")
        except Exception as e:
            print(f"Error al cargar usuarios de Firebase: {e}")
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return
        self.frame_counter += 1
        if self.frame_counter % 20 == 1:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            self.active_trackers = []
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.55)
                name = "Desconocido"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names_db[first_match_index]
                    current_time = time.time()
                    if name not in self.last_spoken_times or (current_time - self.last_spoken_times.get(name, 0) > self.cooldown_period):
                        threading.Thread(target=self._speak_safely, args=(f"He detectado a {name}", App.get_running_app().speech_lock)).start()
                        self.last_spoken_times[name] = current_time
                bbox = (left * 4, top * 4, (right - left) * 4, (bottom - top) * 4)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bbox)
                self.active_trackers.append((tracker, name))
        else:
            pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            try:
                font = ImageFont.truetype("Roboto-Regular.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            new_trackers = []
            for tracker, name in self.active_trackers:
                success, bbox = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in bbox]
                    color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
                    draw.rectangle(((x, y), (x + w, y + h)), outline=color, width=2)
                    draw.text((x + 6, y + h - 25), name.capitalize(), font=font, fill=(255, 255, 255))
                    new_trackers.append((tracker, name))
            self.active_trackers = new_trackers
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.camera_feed.texture = image_texture

    def _speak_safely(self, text, lock):
        with lock:
            local_engine = pyttsx3.init()
            local_engine.say(text)
            local_engine.runAndWait()

    def on_leave(self, *args):
        Clock.unschedule(self.update)
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file('asistente.kv')

class AsistenteApp(App):
    current_user = None
    registered_user = None
    selected_patient_id = None
    speech_lock = threading.Lock()

    def build(self):
        return kv

    def on_stop(self):
        os._exit(0)

if __name__ == '__main__':
    class LoginScreen(Screen): pass
    class UserManagementScreen(Screen): pass
    class FacialLoginScreen(Screen): pass
    class ChoiceScreen(Screen): pass
    class FacialRegistrationScreen(Screen): pass
    class VoiceRegistrationScreen(Screen): pass
    class RoleSelectionScreen(Screen): pass
    class MainMenuScreen(Screen): pass
    class CaregiverDashboardScreen(Screen): pass
    class AddPatientScreen(Screen): pass
    class PatientManagementScreen(Screen): pass
    class RemindersListScreen(Screen): pass
    class OrientationScreen(Screen): pass
    class RecognitionScreen(Screen): pass
    AsistenteApp().run()