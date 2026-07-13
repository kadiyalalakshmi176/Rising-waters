document.addEventListener('DOMContentLoaded', () => {
    // 1. Scroll Progress Bar
    window.addEventListener('scroll', () => {
        const scrollProgress = document.getElementById('scroll-progress');
        if (scrollProgress) {
            const winScroll = document.documentElement.scrollTop || document.body.scrollTop;
            const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            const scrolled = (winScroll / height) * 100;
            scrollProgress.style.width = scrolled + '%';
        }
        
        // Navbar Scrolled Class
        const navbar = document.getElementById('navbar');
        if (navbar) {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        }
        
        // Back to top visibility
        const backToTop = document.getElementById('back-to-top');
        if (backToTop) {
            if (window.scrollY > 300) {
                backToTop.classList.add('show');
            } else {
                backToTop.classList.remove('show');
            }
        }
    });

    // 2. Cursor Glow Tracker
    const cursorGlow = document.getElementById('cursor-glow');
    document.addEventListener('mousemove', (e) => {
        if (cursorGlow) {
            cursorGlow.style.left = e.clientX + 'px';
            cursorGlow.style.top = e.clientY + 'px';
        }
    });

    // 3. Expandable Selected Model details
    const modelCard = document.getElementById('model-card');
    if (modelCard) {
        modelCard.addEventListener('click', () => {
            modelCard.classList.toggle('active');
            const icon = document.getElementById('expand-icon');
            if (icon) {
                icon.textContent = modelCard.classList.contains('active') ? '▲' : '▼';
            }
        });
    }

    // 4. Rain Ambience Audio player
    const rainBtn = document.getElementById('rain-btn');
    const rainAudio = document.getElementById('rain-audio');
    if (rainBtn && rainAudio) {
        rainBtn.addEventListener('click', () => {
            if (rainAudio.paused) {
                rainAudio.play();
                rainBtn.textContent = '⏸ Stop Rain Ambience';
                rainBtn.style.borderColor = 'var(--accent)';
            } else {
                rainAudio.pause();
                rainBtn.textContent = '🔊 Play Rain Ambience';
                rainBtn.style.borderColor = 'var(--glass-border)';
            }
        });
    }

    // 5. Floating Chatbot Panel
    const chatBtn = document.getElementById('chat-btn');
    const chatPanel = document.getElementById('chat-panel');
    const chatClose = document.getElementById('chat-close');
    const chatSend = document.getElementById('chat-send');
    const chatInput = document.getElementById('chat-input');
    const chatMsgs = document.getElementById('chat-msgs');

    if (chatBtn && chatPanel) {
        chatBtn.addEventListener('click', () => {
            chatPanel.style.display = chatPanel.style.display === 'flex' ? 'none' : 'flex';
        });
    }
    if (chatClose) {
        chatClose.addEventListener('click', () => {
            chatPanel.style.display = 'none';
        });
    }

    const appendMessage = (text, sender) => {
        const msg = document.createElement('div');
        msg.classList.add('message', sender);
        msg.textContent = text;
        chatMsgs.appendChild(msg);
        chatMsgs.scrollTop = chatMsgs.scrollHeight;
    };

    const handleChatSubmit = () => {
        const text = chatInput.value.trim();
        if (!text) return;
        appendMessage(text, 'user');
        chatInput.value = '';

        // Simple mock bot replies matching meteorological questions
        setTimeout(() => {
            let reply = "I'm currently processing standard meteorological bounds. Let me know if you need parameters recommendations.";
            const tLower = text.toLowerCase();
            if (tLower.includes('rainfall') || tLower.includes('rain')) {
                reply = "Rainfall above 2000mm annually or 1300mm seasonally indicates strong monsoonal saturations, shifting classification limits.";
            } else if (tLower.includes('xgboost') || tLower.includes('model')) {
                reply = "The deployment model utilizes XGBoost (Extreme Gradient Boosting) yielding a validated accuracy metric of 96.55%.";
            } else if (tLower.includes('humidity') || tLower.includes('temp')) {
                reply = "Humidity above 80% combined with high temperatures increases convective cloud formation chances, raising risks.";
            }
            appendMessage(reply, 'bot');
        }, 600);
    };

    if (chatSend && chatInput) {
        chatSend.addEventListener('click', handleChatSubmit);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleChatSubmit();
        });
    }

    // 6. AI Loading screen sequence overlay (form intercept)
    const form = document.getElementById('predictionForm');
    const overlay = document.getElementById('loading-overlay');
    const fill = document.getElementById('loading-fill');
    const txt = document.getElementById('loading-txt');
    const msg = document.getElementById('loading-msg');

    if (form && overlay) {
        form.addEventListener('submit', (e) => {
            e.preventDefault(); // Stop raw submission
            overlay.style.display = 'flex';
            
            const steps = [
                { pct: 20, txt: "Analyzing Weather...", msg: "Retrieving user-entered parameter floats..." },
                { pct: 40, txt: "Loading AI Model...", msg: "Resolving floods.save binary structures..." },
                { pct: 60, txt: "Checking Rainfall...", msg: "Normalizing annual and seasonal rainfall quantities..." },
                { pct: 80, txt: "Calculating Probability...", msg: "Computing gradient-boosted decision node outputs..." },
                { pct: 100, txt: "Generating Prediction...", msg: "Preparing HTML warnings / stable pages..." }
            ];

            let stepIdx = 0;
            const cycleSteps = () => {
                if (stepIdx < steps.length) {
                    const step = steps[stepIdx];
                    fill.style.width = step.pct + '%';
                    txt.textContent = step.txt;
                    msg.textContent = step.msg;
                    stepIdx++;
                    setTimeout(cycleSteps, 500); // 500ms per step = 2.5 seconds total
                } else {
                    // Finally trigger raw submit
                    form.submit();
                }
            };
            cycleSteps();
        });
    }
});

// 7. Filter History Table
function filterHistoryTable() {
    const input = document.getElementById("search-log");
    if (!input) return;
    const filter = input.value.toUpperCase();
    const table = document.getElementById("history-table");
    if (!table) return;
    const tr = table.getElementsByTagName("tr");

    for (let i = 1; i < tr.length; i++) {
        let match = false;
        const tds = tr[i].getElementsByTagName("td");
        for (let j = 0; j < tds.length; j++) {
            if (tds[j]) {
                const textVal = tds[j].textContent || tds[j].innerText;
                if (textVal.toUpperCase().indexOf(filter) > -1) {
                    match = true;
                    break;
                }
            }
        }
        tr[i].style.display = match ? "" : "none";
    }
}
