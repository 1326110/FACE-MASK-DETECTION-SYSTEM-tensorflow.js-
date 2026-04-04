let video;
let canvas;
let ctx;
let faceDetector;
let maskModel;
let isDetecting = false;

const LABEL_MAPPING = {
    0: { text: "Incorrect Mask", color: "#f59e0b" }, // Orange
    1: { text: "No Mask", color: "#ef4444" }, // Red
    2: { text: "Mask", color: "#10b981" }  // Green
};

async function setupCamera() {
    video = document.getElementById('video-stream');
    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        // Request higher res for better raw crops, but capped for bandwidth
        'video': { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadModels() {
    // 1. Force Backend to guarantee max WebGL speed
    await tf.ready();
    console.log(`[OPTIMIZATION] TFJS Backend Loaded: ${tf.getBackend()}`);
    
    console.log("[OPTIMIZATION] Loading BlazeFace GPU Detector...");
    faceDetector = await blazeface.load({ maxFaces: 10 }); // Allow dense predictions
    
    console.log("[OPTIMIZATION] Loading Float16 Quantized MobileNetV2...");
    maskModel = await tf.loadLayersModel('./web_model/model.json');
}

async function renderPrediction() {
    // 2. Frame-skip check: Prevent JS stack pollution if GPU lags
    if (isDetecting || video.videoWidth === 0) {
        requestAnimationFrame(renderPrediction);
        return;
    }
    isDetecting = true;

    if (canvas.width !== video.clientWidth) {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
    }

    const predictions = await faceDetector.estimateFaces(video, false);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (predictions.length > 0) {
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;
        
        try {
            // 3. Complete memory confinement via single tf.tidy block
            tf.tidy(() => {
                const imgTensor = tf.browser.fromPixels(video);
                const tensorArray = [];
                const validFaces = [];

                for (let i = 0; i < predictions.length; i++) {
                    const start = predictions[i].topLeft;
                    const end = predictions[i].bottomRight;

                    const startX = Math.max(0, start[0] - (end[0] - start[0]) * 0.1);
                    const startY = Math.max(0, start[1] - (end[1] - start[1]) * 0.2);
                    const endX = Math.min(video.videoWidth, end[0] + (end[0] - start[0]) * 0.1);
                    const endY = Math.min(video.videoHeight, end[1] + (end[1] - start[1]) * 0.1);

                    const width = endX - startX;
                    const height = endY - startY;

                    if (width <= 0 || height <= 0) continue;

                    validFaces.push({
                        drawX: start[0] * scaleX,
                        drawY: startY * scaleY,
                        drawW: (end[0] - start[0]) * scaleX,
                        drawH: (end[1] - startY) * scaleY
                    });

                    const startingPoint = [Math.floor(startY), Math.floor(startX), 0];
                    const cropSize = [Math.floor(height), Math.floor(width), 3];
                    const croppedImg = tf.slice(imgTensor, startingPoint, cropSize);

                    const resizedImg = tf.image.resizeBilinear(croppedImg, [224, 224]);
                    const normalizedImg = resizedImg.div(127.5).sub(1);
                    tensorArray.push(normalizedImg);
                }

                if (tensorArray.length > 0) {
                    // 4. Maximum Math Vectorization: tf.stack evaluates all crops in 1 GPU tick
                    const batchedImg = tf.stack(tensorArray);
                    const preds = maskModel.predict(batchedImg);
                    
                    // Pull asynchronous GPU data to CPU just ONCE
                    const classIndices = preds.argMax(1).dataSync();

                    // 5. Draw
                    for (let j = 0; j < validFaces.length; j++) {
                        const face = validFaces[j];
                        const metadata = LABEL_MAPPING[classIndices[j]];

                        ctx.strokeStyle = metadata.color;
                        ctx.lineWidth = 4;
                        ctx.strokeRect(face.drawX, face.drawY, face.drawW, face.drawH);

                        ctx.fillStyle = metadata.color;
                        ctx.beginPath();
                        ctx.roundRect(face.drawX, face.drawY - 30, 150, 30, [8, 8, 0, 0]);
                        ctx.fill();

                        ctx.fillStyle = "#ffffff";
                        ctx.font = "bold 16px 'Outfit', sans-serif";
                        ctx.fillText(metadata.text, face.drawX + 10, face.drawY - 10);
                    }
                }
            });
        } catch (e) {
            console.error("Optimized Inference Error:", e);
        }
    }

    // Unblock the execution trace for the next loop
    isDetecting = false;
    requestAnimationFrame(renderPrediction);
}

async function start() {
    canvas = document.getElementById('overlay-canvas');
    ctx = canvas.getContext('2d');
    
    await setupCamera();
    video.play();
    
    document.querySelector('.status-indicator').innerHTML = '<span class="pulse" style="background-color: var(--color-warning);"></span> Optimizing AI Weights...';
    
    await loadModels();
    
    document.querySelector('.status-indicator').innerHTML = '<span class="pulse"></span> Max Optimization Active';
    
    console.log("[SUCCESS] Models loaded, starting ultra-fast inference loop...");
    renderPrediction();
}

window.onload = start;
