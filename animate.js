
document.getElementById("startBtn").addEventListener("click", () => {
  fetch('/emotion_talk', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
      console.log("Detected Emotion:", data.emotion);
      console.log("Bot says:", data.reply);

      const mouth = document.querySelector('.mouth');
      mouth.style.animation = "none";
      void mouth.offsetWidth;

      if (data.emotion === 'happy') {
        mouth.style.animation = "talk 0.5s ease-in-out infinite";
      } else if (data.emotion === 'sad') {
        mouth.style.height = "20px";
        mouth.style.borderRadius = "0 0 50px 50px";
      } else if (data.emotion === 'surprise') {
        mouth.style.width = "120px";
        mouth.style.height = "80px";
        mouth.style.borderRadius = "50%";
      }

      document.getElementById("botResponse").innerText = data.reply;
      speakText(data.reply);
      animateMouthByPhoneme(data.reply);
      changeFaceEmoji(data.emotion);
    });
});

function speakText(text) {
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'hi-IN';
  utterance.pitch = 1.2;
  utterance.rate = 1.05;
  speechSynthesis.speak(utterance);
}

function animateMouthByPhoneme(text) {
  const mouth = document.querySelector('.mouth');
  const words = text.toLowerCase().split(/\s+/);
  const phonemes = words.flatMap(word => window.phonemeDict[word] || []);
  let i = 0;
  const interval = setInterval(() => {
    if (i >= phonemes.length) {
      clearInterval(interval);
      mouth.style.height = "60px";
      return;
    }
    const p = phonemes[i];
    mouth.style.height = (["AA", "OW", "UW"].includes(p)) ? "80px" : "40px";
    i++;
  }, 200);
}

function changeFaceEmoji(emotion) {
  const face = document.getElementById("emojiFace");
  if (emotion === 'happy') {
    face.style.background = "#00ffff";
    face.style.boxShadow = "0 0 50px #00ffff";
  } else if (emotion === 'sad') {
    face.style.background = "#3380ff";
    face.style.boxShadow = "0 0 50px #3380ff";
  } else if (emotion === 'surprise') {
    face.style.background = "#ffff66";
    face.style.boxShadow = "0 0 50px #ffff66";
  } else {
    face.style.background = "#aaaaaa";
    face.style.boxShadow = "0 0 30px #aaaaaa";
  }
}
