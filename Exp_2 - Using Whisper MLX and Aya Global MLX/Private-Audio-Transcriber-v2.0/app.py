#----------------------
# Private Audio Transcriber (PAT)
# Creator: vbookshelf
# GitHub: https://github.com/vbookshelf/Private-Audio-Transcriber-MLX
# License: MIT
# Version: 2.0 (Translation added)
#----------------------

import os
import sys
import glob
import socket
from flask import Flask, render_template_string, request, jsonify
import re
import json     
import tempfile
import webbrowser
from threading import Timer
import mlx_whisper
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# --- MLX Model Loading ---
print("Loading translation model, please wait...")
try:
    model, tokenizer = load("models/tiny-aya-global-8bit-mlx")
    print("Translation model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load the translation model. Error: {e}")
    print("Please ensure the 'models/tiny-aya-global-8bit-mlx' directory exists and is correct.")
    sys.exit(1)


# --- Language Configuration Logic ---
CONFIG_FILE = "languages-config.txt"
SUPPORTED_LANG_FILE = "supported-languages-aya.json"
DEFAULT_LANGUAGES = [
    "English",
    "French",
    "German",
    "Hindi",
    "Portuguese",
    "Chinese",
    "Spanish",
    "Tamil",
    "Thai",
]

def load_supported_languages():
    """Return the set of officially supported language names from the JSON file."""
    if os.path.exists(SUPPORTED_LANG_FILE):
        with open(SUPPORTED_LANG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("languages", []))
    return set()

def load_languages():
    if not os.path.exists(CONFIG_FILE) or os.stat(CONFIG_FILE).st_size == 0:
        print(f"Config file '{CONFIG_FILE}' not found or empty. Creating with defaults...")
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write("# Add one language per line. Restart app to update.\n")
            for lang in sorted(DEFAULT_LANGUAGES):
                f.write(f"{lang}\n")
        return sorted(DEFAULT_LANGUAGES)
    
    languages = []
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                languages.append(line)
    return sorted(languages) if languages else sorted(DEFAULT_LANGUAGES)

def cleanup_orphaned_temp_files():
    """Delete any audio files left behind by a previous crashed session."""
    upload_dir = os.path.join(os.getcwd(), "temp_user_uploads")
    if not os.path.exists(upload_dir):
        return
    orphans = glob.glob(os.path.join(upload_dir, "*"))
    for f in orphans:
        try:
            os.remove(f)
            print(f"Cleaned up orphaned temp file: {f}")
        except Exception as e:
            print(f"Warning: could not remove orphaned temp file {f}: {e}")

# --- Transcription ---
def run_transcription(audio_path):
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo="models/whisper-turbo-mlx"
    )
    text = result['text'].strip()
    language_code = result['language'] # This is the ISO code (e.g., 'en')
	
    if language_code == 'en':
        dictation_keywords = ['comma', 'period', 'colon', 'new paragraph', 'end of note']
        highlighted_text = text
        for keyword in dictation_keywords:
            pattern = r'\b(' + re.escape(keyword) + r')\b'
            highlighted_text = re.sub(
                pattern,
                lambda match: f"<{match.group(1)}>",
                highlighted_text,
                flags=re.IGNORECASE
            )
        return highlighted_text, language_code
    else:
        return text, language_code

# --- Flask Application ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

@app.after_request
def add_security_headers(response):
    csp = (
        "default-src 'self';"
        "style-src 'self' 'unsafe-inline';"
        "script-src 'self' 'unsafe-inline';"
        "media-src 'self' blob:;"
    )
    response.headers['Content-Security-Policy'] = csp
    response.headers['X-Frame-Options'] = 'DENY'
    # Prevent MIME-type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    # Never send a Referer header — this is a local private app
    response.headers['Referrer-Policy'] = 'no-referrer'
    # Explicitly declare only microphone is used; block everything else
    response.headers['Permissions-Policy'] = 'microphone=(self), camera=(), geolocation=(), payment=()'
    return response

def check_host(host_to_check):
    if host_to_check not in ("127.0.0.1", "localhost"):
        print(f"ERROR: Attempting to bind to a non-local host '{host_to_check}'. Aborting.")
        sys.exit(1)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PAT</title>
	
	<link rel="shortcut icon" type="image/png" href="static/icon.png">
	
    <style>
        :root {
            --primary: #2563eb; --primary-hover: #1d4ed8; --bg: #f1f5f9;
            --card: #ffffff; --text: #1e293b; --text-light: #64748b;
            --danger: #ef4444; --border: #e2e8f0; --edit-bg: #fffcf0; --translate-bg: #f0f9ff;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, sans-serif; }
		
		
        /* Centering the entire app on large displays */
        html {
            background-color: #111827; /* Matches your sidebar footer color */
            display: flex;
            justify-content: center;
            align-items: center;
        }

        body { 
            background-color: var(--bg); 
            color: var(--text); 
            height: 100vh; 
            display: flex; 
            overflow: hidden; 
            
            /* Max width settings */
            width: 100vw;
            max-width: 1440px; 
            margin: 0 auto;
            
            /* Optional: subtle border to define the app edge */
            border-left: 1px solid #374151;
            border-right: 1px solid #374151;
            box-shadow: 0 0 40px rgba(0,0,0,0.5);
        }
		
		
        .left-col { width: 350px; background: #1f2937; border-right: 1px solid #374151; color: #d1d5db; display: flex; flex-direction: column; align-items: stretch; box-shadow: 4px 0 10px rgba(0,0,0,0.2); overflow: hidden; }
        .left-col-scroll { flex: 1; overflow-y: auto; display: flex; flex-direction: column; align-items: center; padding: 2rem 2rem 1rem 2rem; text-align: center; min-height: 0; }
        .left-col-footer { flex-shrink: 0; padding: 0 1rem 1rem 1rem; border-top: 1px solid #374151; background: #1f2937; }
        .right-col { flex: 1; padding: 0.5rem; overflow-y: auto; background: var(--bg); display: flex; flex-direction: column; }
        .mic-container { position: relative; display: flex; flex-direction: column; align-items: center; width: 100%;}
        .mic-btn { width: 100px; height: 100px; border-radius: 50%; border: none; background: var(--primary); color: white; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4); margin-bottom: 1rem; }
        .mic-btn.recording { background: var(--danger); transform: scale(1.1); box-shadow: 0 0 0 15px rgba(239, 68, 68, 0.2); animation: pulse-ring 1.5s infinite; }
        @keyframes pulse-ring { 0% { box-shadow: 0 0 0 0px rgba(239, 68, 68, 0.4); } 100% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); } }
        .hotkey-hint { padding: 0.4rem 0.8rem; background: #374151; border: 1px solid #4b5563; border-radius: 4px; font-size: 0.75rem; color: #d1d5db; display: inline-flex; align-items: center; gap: 0.5rem; }
        kbd { background: #4b5563; border: 1px solid #6b7280; color: #e5e7eb; border-radius: 3px; padding: 1px 6px; font-family: monospace; box-shadow: 0 1px 0 rgba(0,0,0,0.2); }
        .translation-container { width: 100%; margin: 1.5rem 0 0.5rem 0; text-align: left; }
        .translation-container label { font-size: 0.8rem; text-transform: uppercase; color: #9ca3af; margin-bottom: 0.5rem; display: block; }
		
		
        .select-with-button { display: flex; align-items: center; gap: 0.5rem; }
        #language-select { flex-grow: 1; padding: 0.5rem; background: #374151; color: #e5e7eb; border: 1px solid #4b5563; border-radius: 4px; }
		
		#language-select:focus {
		  outline: none; /* Optional: Removes the default browser blue glow */
		}
        
        #remove-lang-btn {
            background: #4b5563; border: 1px solid #6b7280; color: #e5e7eb; border-radius: 4px;
            padding: 6px; cursor: pointer; display: flex; align-items: center; justify-content: center;
            flex-shrink: 0; transition: all 0.2s;
        }
        #remove-lang-btn:hover { background: var(--danger); border-color: #ef4444; }
        #remove-lang-btn svg { width: 16px; height: 16px; }

        /* Search Styles */
        .search-container { width: 100%; position: relative; margin-bottom: 1.5rem; text-align: left; }
        .search-container label { font-size: 0.7rem; text-transform: uppercase; color: #9ca3af; margin-bottom: 0.3rem; display: block; }
		
		
        #lang-search { width: 100%; padding: 0.5rem; background: #111827; color: #e5e7eb; border: 1px solid #4b5563; border-radius: 4px; font-size: 0.85rem; }
		
		#lang-search:focus {
		  outline: none; /* Optional: Removes the default browser blue glow */
		}

		
		
        #search-results { position: absolute; top: 100%; left: 0; right: 0; background: #1f2937; border: 1px solid #4b5563; border-radius: 4px; max-height: 200px; overflow-y: auto; z-index: 100; display: none; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); }
        .search-item { padding: 0.5rem; cursor: pointer; border-bottom: 1px solid #374151; font-size: 0.85rem; }
        .search-item:hover { background: var(--primary); color: white; }

        .separator { font-weight: 600; color: #6b7280; margin: 1.5rem 0; width: 100%; text-align: center; border-bottom: 1px solid #374151; line-height: 0.1em; }
        .separator span { background: #1f2937; padding: 0 10px; }
        #drop-zone { width: 100%; border: 2px dashed #4b5563; border-radius: 8px; padding: 1rem; text-align: center; cursor: pointer; transition: background-color 0.2s, border-color 0.2s; }
        #drop-zone.drag-over { background-color: #374151; border-color: var(--primary); }
        #drop-zone p { color: #9ca3af; margin-top: 0.5rem; font-size: 0.8rem; }
        #file-input { display: none; }
        #file-list-container { width: 100%; margin-top: 0.75rem; text-align: left; }
        #file-list-container h3 { font-size: 0.8rem; text-transform: uppercase; color: #9ca3af; margin-bottom: 0.5rem; }
        #file-list { list-style: none; max-height: 110px; overflow-y: auto; background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 0.5rem; }
        #file-list li { font-size: 0.85rem; padding: 0.4rem 0.6rem; border-bottom: 1px solid #374151; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .status-loader-row { display: flex; align-items: center; gap: 0.75rem; margin-top: 0.75rem; width: 100%; justify-content: center; min-height: 30px; }
        .status-text { font-weight: 600; font-size: 0.9rem; color: #9ca3af; }
        .loader { display: none; flex-shrink: 0; border: 3px solid #f3f3f3; border-top: 3px solid var(--primary); border-radius: 50%; width: 22px; height: 22px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .right-header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 1rem; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border); max-width: 800px; width: 100%; margin-left: auto; margin-right: auto; }
        .header-note { font-style: italic; font-size: 0.9rem; color: var(--text-light); }
        .header-actions { display: flex; gap: 1rem; }
        .chart-paper { background: white; padding: 2rem 3rem; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); max-width: 800px; margin: 0 auto; width: 100%; flex-grow: 1; }
        .transcription-group { margin-bottom: 2.5rem; }
        .field-label { font-size: 0.75rem; font-weight: 700; color: var(--primary); text-transform: uppercase; margin-bottom: 0.5rem; display: block; }
        .text-area-with-actions { position: relative; width: 100%; display: flex; align-items: flex-start; gap: 0.5rem; }
        .text-area-wrapper { position: relative; width: 100%; flex-grow: 1;}
        .copy-btn { position: absolute; top: 8px; right: 8px; background: white; border: 1px solid var(--border); border-radius: 4px; padding: 5px; cursor: pointer; z-index: 5; display: flex; align-items: center; justify-content: center; transition: all 0.2s; opacity: 0.6; }
        .copy-btn:hover { opacity: 1; background: #f8fafc; border-color: var(--primary); }
        .copy-btn svg { width: 16px; height: 16px; color: var(--text-light); }
        .transcription-area, .translation-textarea { width: 100%; padding: 0.75rem; padding-right: 2.5rem; border-radius: 0.4rem; border: 1px solid #e2e8f0; line-height: 1.8; color: var(--text); outline: none; transition: all 0.2s; resize: vertical; font-size: 1rem; transition: height 0.2s ease-in-out; }
        .transcription-area { min-height: 250px; background: #fafafa; }
        .transcription-area:focus { background: var(--edit-bg); border-color: #fbbf24; box-shadow: 0 0 0 2px rgba(251, 191, 36, 0.3); }
        .action-buttons { display: flex; flex-direction: column; gap: 0.5rem; }
        .action-btn { background: white; border: 1px solid var(--border); border-radius: 4px; padding: 5px; cursor: pointer; z-index: 5; display: flex; align-items: center; justify-content: center; transition: all 0.2s; opacity: 0.6; }
        .action-btn:hover { opacity: 1; background: #f8fafc; border-color: var(--primary); }
        .action-btn svg { width: 16px; height: 16px; color: var(--text-light); }
        .translation-output { margin-top: 1rem; }
        .translation-textarea { min-height: 150px; background: var(--translate-bg); border-color: #93c5fd; color: #075985; }
        .translation-textarea:focus { background: #e0f2fe; border-color: #38bdf8; box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.3); }
        .btn-small { padding: 0.5rem 1rem; font-size: 0.85rem; border-radius: 4px; border: 1px solid var(--border); background: white; cursor: pointer; display: flex; align-items: center; gap: 0.4rem; }
        .privacy-notice { font-size: 0.7rem; color: #6b7280; background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 0.5rem 0.75rem; margin-top: 0.75rem; width: 100%; text-align: left; line-height: 1.4; }
        .privacy-notice strong { color: #9ca3af; display: block; margin-bottom: 0.2rem; }
    </style>
</head>
<body>
    <aside class="left-col">
        <div class="left-col-scroll">
            <div class="mic-container">
                <button id="micBtn" class="mic-btn">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
                </button>
                <div class="hotkey-hint"><kbd>Space</kbd> Start / Stop Recording</div>
            </div>

            <div class="translation-container">
                <label for="language-select">Translate To:</label>
                <div class="select-with-button">
				
				<select id="language-select">
				    {% for lang in languages %}
				    <option value="{{ lang }}" {% if lang == 'English' %}selected{% endif %}>{{ lang }}</option>
				    {% endfor %}
				</select>
					
                    <button id="remove-lang-btn" title="Remove selected language">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    </button>
                </div>
            </div>

            <div class="search-container">
                <!--
                <label for="lang-search">Search & Add Languages:</label>
                -->
                <input type="text" id="lang-search" placeholder="Add a language..." autocomplete="off">
                <div id="search-results"></div>
            </div>

            <!--
            <div class="separator"><span>-x-</span></div>
            -->

            <label for="file-input" id="drop-zone">
                <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                <p>Click or drag audio files.<br>wav - mp3 - m4a</p>
            </label>
            <input type="file" id="file-input" multiple accept="audio/*">

            <div id="file-list-container" style="display: none;">
                <h3>Processed Files</h3>
                <ul id="file-list"></ul>
            </div>

            <div class="status-loader-row">
                <div id="loader" class="loader"></div>
                <div id="statusText" class="status-text">Ready</div>
            </div>
        </div>

        <div class="left-col-footer">
            <div class="privacy-notice">
                <strong>🔒 Privacy Notice</strong>
                Transcriptions are temporarily held in browser session storage and are cleared when this tab is closed.
            </div>
        </div>
    </aside>

    <main class="right-col">
        <div class="right-header">
            <span class="header-note">AI can make mistakes. Please double-check.</span>
            <div class="header-actions">
                <button class="btn-small" onclick="clearSession()">Reset Form</button>
            </div>
        </div>
        <div class="chart-paper">
            <div id="results-container"></div>
        </div>
    </main>

    <script>
        const micBtn = document.getElementById('micBtn');
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const fileListContainer = document.getElementById('file-list-container');
        const fileList = document.getElementById('file-list');
        const statusText = document.getElementById('statusText');
        const loader = document.getElementById('loader');
        const resultsContainer = document.getElementById('results-container');
        const langSearch = document.getElementById('lang-search');
        const searchResults = document.getElementById('search-results');
        const languageSelect = document.getElementById('language-select');
        const removeLangBtn = document.getElementById('remove-lang-btn');

        const COPY_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
        const CHECK_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
        const TRANSLATE_ICON = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h20M12 2a10 10 0 110 20 10 10 0 010-20z"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>`;

        let isRecording = false;
        let mediaRecorder = null;
        let audioChunks = [];
        let audioBlob = null;
        // FIX: supportedLanguages is an array of plain strings (language names),
        // not objects with .language / .code properties.
        let supportedLanguages = [];

        // Load supported languages for search
        fetch('/get_supported_languages')
            .then(res => res.json())
            // FIX: The server returns { "languages": [...] }, so read .languages
            .then(data => { supportedLanguages = data.languages || []; });

        langSearch.addEventListener('input', (e) => {
            const val = e.target.value.toLowerCase();
            searchResults.innerHTML = '';
            if (!val) { searchResults.style.display = 'none'; return; }
            
            // FIX: supportedLanguages is an array of strings, so filter/compare directly
            const filtered = supportedLanguages.filter(lang =>
                lang.toLowerCase().startsWith(val)
            ).slice(0, 10);

            if (filtered.length > 0) {
                filtered.forEach(lang => {
                    const div = document.createElement('div');
                    div.className = 'search-item';
                    // FIX: lang is already a plain string — use it directly
                    div.textContent = lang;
                    div.onclick = () => addLanguage(lang);
                    searchResults.appendChild(div);
                });
                searchResults.style.display = 'block';
            } else {
                searchResults.style.display = 'none';
            }
        });

        async function addLanguage(langStr) {
            langSearch.value = '';
            searchResults.style.display = 'none';
            
            // Check if already in dropdown
            const exists = Array.from(languageSelect.options).some(opt => opt.value === langStr);
            if (exists) {
                languageSelect.value = langStr;
                return;
            }

            try {
                const response = await fetch('/add_language', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'MedicalApp' },
                    body: JSON.stringify({ language: langStr })
                });
                if (response.ok) {
                    const opt = document.createElement('option');
                    opt.value = langStr;
                    opt.textContent = langStr;
                    languageSelect.appendChild(opt);
                    
                    // Sort dropdown
                    const options = Array.from(languageSelect.options);
                    options.sort((a, b) => a.text.localeCompare(b.text));
                    languageSelect.innerHTML = '';
                    options.forEach(o => languageSelect.add(o));
                    
                    languageSelect.value = langStr;
                }
            } catch (e) { console.error("Failed to save language", e); }
        }

        async function removeSelectedLanguage() {
            const selectedOption = languageSelect.options[languageSelect.selectedIndex];
            if (!selectedOption) {
                alert("No language selected to remove.");
                return;
            }

            const langToRemove = selectedOption.value;
            
            if (languageSelect.options.length <= 1) {
                alert("You cannot remove the last language.");
                return;
            }

            if (confirm(`Are you sure you want to remove "${langToRemove}" from the list?`)) {
                try {
                    const response = await fetch('/remove_language', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'MedicalApp' },
                        body: JSON.stringify({ language: langToRemove })
                    });

                    if (response.ok) {
                        selectedOption.remove();
                        languageSelect.selectedIndex = 0;
                    } else {
                        const errData = await response.json();
                        alert(`Failed to remove language: ${errData.error}`);
                    }
                } catch (e) {
                    console.error("Failed to remove language", e);
                    alert("An error occurred while trying to remove the language.");
                }
            }
        }

        removeLangBtn.addEventListener('click', removeSelectedLanguage);

        // Close search results when clicking outside
        document.addEventListener('click', (e) => {
            if (!langSearch.contains(e.target)) searchResults.style.display = 'none';
        });

        window.addEventListener('DOMContentLoaded', () => {
            const savedData = sessionStorage.getItem('transcriptions');
            if (savedData) {
                const transcriptions = JSON.parse(savedData);
                transcriptions.reverse().forEach(item => {
                    displayTranscription(item.fileName, item.text, null, item.id, item.translation, item.source_lang_code);
                });
            }
        });

        function saveToSession() {
            const groups = document.querySelectorAll('.transcription-group');
            const dataToSave = Array.from(groups).map(group => {
                const translationTextarea = group.querySelector('textarea.translation-textarea');
                return {
                    id: group.dataset.id,
                    fileName: group.querySelector('.field-label').textContent,
                    text: group.querySelector('textarea.transcription-area').value,
                    translation: translationTextarea ? translationTextarea.value : null,
                    source_lang_code: group.dataset.sourceLangCode
                };
            });
            sessionStorage.setItem('transcriptions', JSON.stringify(dataToSave));
        }

        function clearSession() {
            if(confirm("Are you sure you want to clear all transcriptions?")) {
                sessionStorage.removeItem('transcriptions');
                location.reload();
            }
        }

        window.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
        window.addEventListener('dragleave', (e) => { if (!e.relatedTarget) { dropZone.classList.remove('drag-over'); } });
        window.addEventListener('drop', (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); handleFiles(e.dataTransfer.files); });
        fileInput.addEventListener('change', () => handleFiles(fileInput.files));

        async function handleFiles(files) {
            if (isRecording) { alert("Please stop the recording before uploading files."); return; }
            if (files.length === 0) return;
            fileListContainer.style.display = 'block';
            loader.style.display = 'block';
            for (const file of [...files]) {
                const li = document.createElement('li');
                li.textContent = `Processing: ${file.name}...`;
                fileList.appendChild(li);
                statusText.innerText = `Transcribing ${file.name}...`;
                await processSingleAudio(file, file.name, file);
                li.textContent = `✓ ${file.name}`;
            }
            statusText.innerText = "Processing complete.";
            loader.style.display = 'none';
            fileInput.value = ''; 
        }

        async function toggleRecording() {
            if (isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                micBtn.classList.remove('recording');
            } else {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    fileInput.value = ''; 
                    isRecording = true;
                    micBtn.classList.add('recording');
                    statusText.innerText = "RECORDING LIVE";
                    audioChunks = [];
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                    mediaRecorder.onstop = async () => {
                        audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        stream.getTracks().forEach(track => track.stop());
                        loader.style.display = 'block';
                        statusText.innerText = "Transcribing recording...";
                        await processSingleAudio(audioBlob, "Live Recording", audioBlob);
                        statusText.innerText = "Ready for next note";
                        loader.style.display = 'none';
                    };
                    mediaRecorder.start();
                } catch (e) { alert("Microphone access denied."); }
            }
        }
        
        micBtn.addEventListener('click', toggleRecording);
        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && e.target.tagName !== 'TEXTAREA') {
                e.preventDefault();
                toggleRecording();
            }
        });

        async function processSingleAudio(audioSource, sourceName, fileObject) {
		    const formData = new FormData();
		    formData.append("audio_file", audioSource, `${sourceName}.webm`);
		    try {
		        const response = await fetch("/transcribe", { 
		            method: "POST", 
		            body: formData,
		            headers: { "X-Requested-With": "MedicalApp" } 
		        });
		        const data = await response.json();
		        displayTranscription(sourceName, data.transcription || 'Could not transcribe.', fileObject, null, null, data.source_lang_code);
		    } catch (error) {
		        displayTranscription(sourceName, 'ERROR: Transcription failed.', fileObject);
		    }
		}

        function displayTranscription(fileName, transcriptionText, fileObject = null, existingId = null, existingTranslation = null, sourceLangCode = 'auto') {
            const uniqueId = existingId || Date.now() + Math.random().toString(36).substr(2, 9);
            const group = document.createElement('div');
            group.className = 'transcription-group';
            group.dataset.id = uniqueId;
            group.dataset.sourceLangCode = sourceLangCode;
            const label = document.createElement('span');
            label.className = 'field-label';
            label.textContent = fileName;
            group.appendChild(label);
            if (fileObject) {
                const audioPlayer = document.createElement('audio');
                audioPlayer.controls = true;
                audioPlayer.src = URL.createObjectURL(fileObject);
                audioPlayer.style.width = '100%';
                audioPlayer.style.marginBottom = '0.75rem';
                group.appendChild(audioPlayer);
            }
            const container = document.createElement('div');
            container.className = 'text-area-with-actions';
            const wrapper = document.createElement('div');
            wrapper.className = 'text-area-wrapper';
            const textarea = document.createElement('textarea');
            textarea.className = 'transcription-area';
            textarea.value = transcriptionText;
            textarea.id = `textarea-${uniqueId}`;
            textarea.oninput = () => saveToSession();
            textarea.onfocus = () => { textarea.style.height = 'auto'; textarea.style.height = (textarea.scrollHeight) + 'px'; };
            textarea.onblur = () => { textarea.style.height = '250px'; };
            const mainCopyBtn = document.createElement('button');
            mainCopyBtn.className = 'copy-btn';
            mainCopyBtn.innerHTML = COPY_ICON;
            mainCopyBtn.onclick = () => {
                navigator.clipboard.writeText(textarea.value);
                mainCopyBtn.innerHTML = CHECK_ICON;
                setTimeout(() => { mainCopyBtn.innerHTML = COPY_ICON; }, 2000);
            };
            wrapper.appendChild(textarea);
            wrapper.appendChild(mainCopyBtn);
            const actionButtons = document.createElement('div');
            actionButtons.className = 'action-buttons';
            const translateBtn = document.createElement('button');
            translateBtn.className = 'action-btn';
            translateBtn.innerHTML = TRANSLATE_ICON;
            translateBtn.title = "Translate text";
            translateBtn.onclick = () => handleTranslate(uniqueId);
            actionButtons.appendChild(translateBtn);
            container.appendChild(wrapper);
            container.appendChild(actionButtons); 
            group.appendChild(container);
            const translationOutput = document.createElement('div');
            translationOutput.id = `translation-${uniqueId}`;
            translationOutput.className = 'translation-output';
            group.appendChild(translationOutput);
            if (existingTranslation) { renderTranslationUI(translationOutput, existingTranslation); }
            resultsContainer.prepend(group);
            saveToSession();
        }
        
        function renderTranslationUI(outputDiv, translationText) {
            outputDiv.innerHTML = '';
            const wrapper = document.createElement('div');
            wrapper.className = 'text-area-wrapper';
            const textarea = document.createElement('textarea');
            textarea.className = 'translation-textarea';
            textarea.value = translationText;
            textarea.oninput = () => saveToSession();
            textarea.onfocus = () => { textarea.style.height = 'auto'; textarea.style.height = (textarea.scrollHeight) + 'px'; };
            textarea.onblur = () => { textarea.style.height = '150px'; };
            const copyBtn = document.createElement('button');
            copyBtn.className = 'copy-btn';
            copyBtn.innerHTML = COPY_ICON;
            copyBtn.onclick = () => {
                navigator.clipboard.writeText(textarea.value);
                copyBtn.innerHTML = CHECK_ICON;
                setTimeout(() => { copyBtn.innerHTML = COPY_ICON; }, 2000);
            };
            wrapper.appendChild(textarea);
            wrapper.appendChild(copyBtn);
            outputDiv.appendChild(wrapper);
        }

        async function handleTranslate(uniqueId) {
            const textToTranslate = document.getElementById(`textarea-${uniqueId}`).value;
            const targetLanguage = document.getElementById('language-select').value;
            const sourceLangCode = document.querySelector(`[data-id="${uniqueId}"]`).dataset.sourceLangCode;
            const outputDiv = document.getElementById(`translation-${uniqueId}`);
            if (!textToTranslate.trim()) { alert("There is no text to translate."); return; }
            outputDiv.innerHTML = `<div class="loader loader-small" style="display: block; border: 2px solid #f3f3f3; border-top: 2px solid var(--primary); width: 16px; height: 16px; margin: 8px; border-radius: 50%; animation: spin 1s linear infinite;"></div>`;
            try {
                const response = await fetch("/translate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json", "X-Requested-With": "MedicalApp" },
                    body: JSON.stringify({ 
                        text: textToTranslate, 
                        language: targetLanguage,
                        source_lang_code: sourceLangCode
                    })
                });
                if (!response.ok) { const errData = await response.json(); throw new Error(errData.error || "Translation request failed."); }
                const data = await response.json();
                renderTranslationUI(outputDiv, data.translation.trim());
                saveToSession();
            } catch(error) {
                outputDiv.innerHTML = `<textarea class="translation-textarea" readonly>Error: ${error.message}</textarea>`;
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    langs = load_languages()
    return render_template_string(HTML_TEMPLATE, languages=langs)

@app.route("/get_supported_languages")
def get_supported_languages():
    if os.path.exists(SUPPORTED_LANG_FILE):
        with open(SUPPORTED_LANG_FILE, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"languages": []})

@app.route("/add_language", methods=["POST"])
def add_language():
    if request.headers.get("X-Requested-With") != "MedicalApp":
        return jsonify({"error": "Unauthorized request source"}), 403

    new_lang = request.json.get('language')
    if not new_lang:
        return jsonify({"error": "No language provided"}), 400

    # Issue #3: Validate against the official supported languages list
    # to prevent arbitrary strings being written to disk
    supported = load_supported_languages()
    if new_lang not in supported:
        return jsonify({"error": "Unsupported language"}), 400

    # Check if already in config to avoid duplicates
    existing = load_languages()
    if new_lang not in existing:
        with open(CONFIG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{new_lang}\n")
    return jsonify({"status": "success"})

@app.route("/remove_language", methods=["POST"])
def remove_language():
    if request.headers.get("X-Requested-With") != "MedicalApp":
        return jsonify({"error": "Unauthorized request source"}), 403

    lang_to_remove = request.json.get('language')
    if not lang_to_remove:
        return jsonify({"error": "No language provided"}), 400

    current_langs = load_languages()
    if lang_to_remove in current_langs:
        updated_langs = [lang for lang in current_langs if lang != lang_to_remove]
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                f.write("# Add one language per line. Restart app to update.\n")
                for lang in sorted(updated_langs):
                    f.write(f"{lang}\n")
            return jsonify({"status": "success"})
        except Exception as e:
            print(f"Config write error: {e}")  # Full details stay server-side only
            return jsonify({"error": "Failed to update config file."}), 500
    else:
        return jsonify({"status": "success", "message": "Language not found in config"})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if request.headers.get("X-Requested-With") != "MedicalApp":
        return jsonify({"error": "Unauthorized request source"}), 403
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files['audio_file']
    upload_dir = os.path.join(os.getcwd(), "temp_user_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    temp_audio_path = None

    # Issue #4: Whitelist extensions — never trust the client-supplied filename
    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac'}
    raw_suffix = os.path.splitext(audio_file.filename)[1].lower()
    suffix = raw_suffix if raw_suffix in ALLOWED_EXTENSIONS else '.webm'

    try:
        with tempfile.NamedTemporaryFile(dir=upload_dir, delete=False, suffix=suffix) as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        transcribed_text, lang_code = run_transcription(temp_audio_path)
        return jsonify({
            "transcription": transcribed_text,
            "source_lang_code": lang_code
        })
    except Exception as e:
        print(f"Transcription error: {e}")  # Full details stay server-side only
        return jsonify({"error": "Transcription failed. Please try again."}), 500
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path): 
            os.remove(temp_audio_path)

@app.route("/translate", methods=["POST"])
def translate():
    if request.headers.get("X-Requested-With") != "MedicalApp":
        return jsonify({"error": "Unauthorized request source"}), 403
    
    data = request.json
    text_to_translate = data.get('text')
    target_lang = data.get('language', '').strip()

    # Aya uses a simple prompt format
    prompt = f"""
Please translate this text into {target_lang}: {text_to_translate}

Output your response as json with the following keys: translation

"""

    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
    
    try:
		
        sampler = make_sampler(temp=0.0)
		
        response_text = generate(model, tokenizer, prompt=prompt, verbose=False, sampler=sampler)
        
        # FIX: .split('```json')[1].split('```') returns a list, not a string.
        # Use indexing to get the content between the fences.
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        
        response_text = response_text.strip()

        # Try to parse the cleaned text as JSON
        try:
            parsed_json = json.loads(response_text)
            translation = parsed_json.get('translation', 'Error: "translation" key not found in model response.')
        except json.JSONDecodeError:
            # If it's not valid JSON, use the raw response as a fallback
            translation = response_text

        return jsonify({"translation": translation.strip()})
    except Exception as e:
        print(f"Translation error: {e}")  # Full details stay server-side only
        return jsonify({"error": "Translation failed. Please try again."}), 500

def open_browser(host, port):
    webbrowser.open_new(f'http://{host}:{port}')

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5001
    check_host(host)
    load_languages()
    cleanup_orphaned_temp_files()
    Timer(1, lambda: open_browser(host, port)).start()
    app.run(host=host, port=port, debug=False)
