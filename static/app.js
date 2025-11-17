// DOM elements
const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat");

// Render message
function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = "message " + (sender === "user" ? "user-msg" : "bot-msg");
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Add typing indicator
function addTyping() {
  const typingDiv = document.createElement("div");
  typingDiv.className = "message bot-msg typing";
  typingDiv.id = "typing";
  typingDiv.innerHTML = "<span></span><span></span><span></span>";
  messagesEl.appendChild(typingDiv);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Remove typing indicator
function removeTyping() {
  const t = document.getElementById("typing");
  if (t) t.remove();
}

// Send message to FastAPI
async function sendMessage() {
  const msg = inputEl.value.trim();
  if (!msg) return;
  inputEl.value = "";

  addMessage(msg, "user");
  addTyping();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg })
    });

    const data = await res.json();
    removeTyping();

    if (data && data.reply) {
      addMessage(data.reply, "bot");
    } else {
      addMessage("⚠️ Server error", "bot");
    }
  } catch (e) {
    removeTyping();
    addMessage("⚠️ Cannot connect to server", "bot");
  }
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keypress", e => {
  if (e.key === "Enter") sendMessage();
});

newChatBtn.addEventListener("click", () => {
  messagesEl.innerHTML = "";
  addMessage("New chat started! 👋", "bot");
});
