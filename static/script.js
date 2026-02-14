document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('generateForm');
    const generateBtn = document.getElementById('generateBtn');
    const resultTab = document.getElementById('resultTab');
    const logsTab = document.getElementById('logsTab');
    const logsConsole = document.getElementById('logsConsole');
    const resultContent = document.getElementById('resultContent');
    const emptyState = document.getElementById('emptyState');
    const generatedTextPre = document.getElementById('generatedText');
    const mediaContainer = document.getElementById('mediaContainer');

    // Tab Switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`${btn.dataset.tab}Tab`).classList.add('active');
        });
    });

    // File Upload UX
    ['reference_image', 'video_init_image'].forEach(id => {
        const input = document.getElementById(id);
        const area = input.parentElement;
        
        area.addEventListener('click', () => input.click());
        
        input.addEventListener('change', () => {
            if (input.files.length) {
                area.querySelector('.file-msg').textContent = input.files[0].name;
                area.style.borderColor = 'var(--success-color)';
            }
        });
    });

    function log(message, type = 'info') {
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logsConsole.appendChild(entry);
        logsConsole.scrollTop = logsConsole.scrollHeight;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Reset UI
        generateBtn.classList.add('generating');
        generateBtn.disabled = true;
        emptyState.style.display = 'block';
        resultContent.classList.add('hidden');
        mediaContainer.innerHTML = '';
        generatedTextPre.textContent = '';
        log('Starting generation process...', 'system');

        const formData = new FormData(form);

        try {
            log('Sending request to agent...');
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            log('Response received.');

            if (data.errors && data.errors.length > 0) {
                data.errors.forEach(err => log(err, 'error'));
                throw new Error('Agent reported errors.');
            }

            // Display Results
            log('Rendering results...', 'success');
            emptyState.style.display = 'none';
            resultContent.classList.remove('hidden');

            // Text
            if (data.generated_text) {
                generatedTextPre.textContent = data.generated_text;
            } else {
                generatedTextPre.textContent = "No text generated.";
            }

            // Media
            if (data.generated_media_url) {
                if (Array.isArray(data.generated_media_url)) {
                    // Carousel
                    data.generated_media_url.forEach(url => {
                        const img = document.createElement('img');
                        img.src = url;
                        mediaContainer.appendChild(img);
                    });
                } else {
                    const url = data.generated_media_url;
                    // Simple check for video extension or if it came from video intent
                    // A more robust way would be checking file extension or content-type
                    const isVideo = url.includes('.mp4') || url.includes('.webm') || formData.get('user_media_choice') === 'video';
                    
                    if (isVideo) {
                        const video = document.createElement('video');
                        video.src = url;
                        video.controls = true;
                        video.autoplay = true;
                        video.loop = true;
                        mediaContainer.appendChild(video);
                    } else {
                        const img = document.createElement('img');
                        img.src = url;
                        mediaContainer.appendChild(img);
                    }
                }
            } else {
                const p = document.createElement('p');
                p.textContent = "No media generated.";
                mediaContainer.appendChild(p);
            }

            log('Process completed successfully.', 'success');

        } catch (error) {
            log(`Error: ${error.message}`, 'error');
            alert('Generation failed. Check logs for details.');
        } finally {
            generateBtn.classList.remove('generating');
            generateBtn.disabled = false;
        }
    });

    window.copyText = () => {
        const text = generatedTextPre.textContent;
        navigator.clipboard.writeText(text).then(() => {
            const btn = document.querySelector('.copy-btn');
            const original = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = original, 2000);
        });
    };
});
