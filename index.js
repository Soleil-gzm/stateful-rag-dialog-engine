const express = require("express");
const https = require("https");
const fs = require("fs");
const bodyParser = require("body-parser");
const cors = require("cors");
const path = require("path");
const { Buffer } = require("buffer");
const { spawn } = require("child_process"); // To run Python scripts
const app = express();
// const port = 5917;
const port = 5902;

// Middleware
// app.use(cors({ origin: "http://172.16.9.53:5500" }));
app.use(cors());
app.use(bodyParser.json());
// app.use(express.json());
// Serve static files from the public directory
app.use(express.static(path.join(__dirname, "public")));
// Increase the limit to 50MB (or adjust as needed)
app.use(express.json({ limit: "10mb" }));
app.use((req, res, next) => {
  const bodySize = Buffer.byteLength(JSON.stringify(req.body));
  // console.log(`Request body size: ${bodySize} bytes`);
  next();
});

// Map to keep track of active listeners by a unique key (chatId + questionId)
let activeListeners = new Map();

function startPythonModels(gpuIndex, memoryFraction) {
  console.log("Starting Python LLM process...");
  const proc = spawn("python3", [
    "run_module.py", 
    "--args_file=src/args/qwen.json", 
    `--kv_cache_free_gpu_memory_fraction=${memoryFraction.toFixed(2)}`
  ], {
    stdio: ["pipe", "pipe", "pipe"], // Redirect stderr to ignore
    env: {
      ...process.env,
      CUDA_VISIBLE_DEVICES: String(gpuIndex), // restrict this process to one GPU
    },
  });

  setupLLMProcess(proc, activeListeners);
  return proc;
}

function handleLLMOutput(data, activeListeners) {
  try {
    const lines = data.toString("utf-8").split("\n");
    for (const line of lines) {
      if (!line.trim()) continue;
      // Skip if the line doesn't look like JSON (starts with {"task_id")
      if (!line.trim().startsWith(`{"task_id"`))  {
        continue;
      } 
      const { task_id: questionId, response: response } = JSON.parse(line);
      // find the matching listener
      const listenerKey = Array.from(activeListeners.keys()).find((key) =>
        key.endsWith(`_${questionId}`)
      );
      if (!listenerKey) {
        console.error(`No active listener for questionId ${questionId}`);
        continue;
      }

      const { res } = activeListeners.get(listenerKey);
      if (!res) continue;

      if (response.startsWith("<END_OF_STREAMING_SIGNAL>")) {
        const dataString = response
          .replace("<END_OF_STREAMING_SIGNAL>", "")
          .trim();
        const [state, node, repeat, check_count, query_node] =
          dataString.split("|");
        const eventData = `state@ ${state} | node@ ${node} | repeat@ ${repeat} | check_count@ ${check_count} | query_node@ ${query_node}`;
        
        console.log(
          new Date().toISOString(),
          `Python LLM finished for questionId ${questionId}; state: ${state}, node: ${node}, repeat: ${repeat}, check_count: ${check_count}, query_node: ${query_node}`
        );
        res.write(`event: end\ndata: ${eventData}\n\n`);
      } else {
        res.write(`data: ${response}\n\n`);
      }
    }
  } catch (err) {
    console.error("Failed to parse output from Python LLM:", err);
  }
}

function setupLLMProcess(llmProcess, activeListeners) {
  llmProcess.stdout.on("data", (data) =>
    handleLLMOutput(data, activeListeners)
  );
  llmProcess.stderr.on("data", (data) => console.error(`LLM Error: ${data}`));
  llmProcess.on("close", (code) => {
    console.log(`LLM exited with code ${code}`);
    llmProcess = null; // ...restart logic if you want...
  });
}

// Custom GPU list to use.
const gpuIndices = [0];
const fractions = [0.1]; // Adjusted fractions for each GPU

// const gpuIndices = [5,5];
// const fractions = [0.01, 0.02]; // Adjusted fractions for each GPU

// Spawn one worker per listed GPU
// const workers = gpuIndices.map((gpuIndex) => startPythonModels(gpuIndex));
const workers = gpuIndices.map((gpuIndex, i) => 
  startPythonModels(gpuIndex, fractions[i])
);

// keep a pointer (0-based) to the next worker
let nextWorkerIdx = 0;

function getNextWorker() {
  const worker = workers[nextWorkerIdx];
  // advance & wrap
  nextWorkerIdx = (nextWorkerIdx + 1) % workers.length;
  return worker;
}

// API endpoint for generating responses with chat history
app.post("/api/llm/generate", async (req, res) => {
  const { chatId, questionId, chatHistory, prompt } = req.body;
  // console.log("prompt is: ", prompt);
  const listenerKey = `${chatId}_${questionId}`;
  console.log(
    new Date().toISOString(),
    `Received new questions with chatId ${chatId} and questionId ${questionId}`
  );

  if (!workers) {
    return res.status(500).json({ error: "LLM is not running." });
  }

  // Add listener for the current question and store chatHistory and prompt
  activeListeners.set(listenerKey, { res: null, chatHistory, prompt }); // Set res to null initially until streaming is accessed

  res.json({ message: "Input received!" });
});

// General route to handle streaming responses for each chatId and questionId
app.get("/api/llm/:chatId/:questionId/stream", (req, res) => {
  const { chatId, questionId } = req.params;
  const listenerKey = `${chatId}_${questionId}`;

  // console.log(
  //   new Date().toISOString(),
  //   `Streaming route accessed for chatId: ${chatId}, questionId: ${questionId}`
  // );

  // Set up response headers for Server-Sent Events (SSE)
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  // Store the listener response object
  if (activeListeners.has(listenerKey)) {
    const listener = activeListeners.get(listenerKey);
    listener.res = res;
    // Send input text to Python once the listener is ready
    const worker = getNextWorker();
    worker.stdin.write(
      `${JSON.stringify({
        task_id: questionId,
        chatHistory: listener.chatHistory,
        prompt: listener.prompt,
      })}\n`
    );
    console.log(
      new Date().toISOString(),
      `Sent request to python LLM successfully: chatId: ${chatId}, questionId: ${questionId}`
    );
  } else {
    console.error(`No active listener found for key: ${listenerKey}`);
    res.status(404).json({ error: "Listener not found." });
    return;
  }

  // Remove listener and cleanup when the response ends unexpectedly
  req.on("close", () => {
    // console.log(
    //   `Request closed unexpectedly for chatId: ${chatId}, questionId: ${questionId}`
    // );
    activeListeners.delete(listenerKey); // Remove listener reference from the Map
    res.end(); // Close the response
  });
});

function terminateChildProcesses() {
  workers.forEach((proc, idx) => {
    if (proc && !proc.killed) {
      proc.kill("SIGTERM"); // ask it to shut down
      console.log(`LLM worker #${idx} (pid ${proc.pid}) terminated.`);
    }
  });
}

// Capture signals and terminate child processes properly
process.on("SIGINT", () => {
  console.log("SIGINT received. Terminating child processes...");
  terminateChildProcesses();
  process.exit(); // Exit the main process
});

process.on("SIGTERM", () => {
  console.log("SIGTERM received. Terminating child processes...");
  terminateChildProcesses();
  process.exit(); // Exit the main process
});

// Optional: handle exit event to ensure cleanup
process.on("exit", () => {
  console.log("Node.js process exiting. Cleaning up child processes...");
  terminateChildProcesses();
});

// Define your HTTPS certificate and key
const options = {
  key: fs.readFileSync("./server.key"), // Path to your private key
  cert: fs.readFileSync("./server.crt"), // Path to your certificate
};

https.createServer(options, app).listen(port, "0.0.0.0", () => {
  console.log(`Server running on https://localhost:${port}`);
});
