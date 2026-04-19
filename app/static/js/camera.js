(() => {
  const MAX_CAPTURE_WIDTH = 1024;
  const JPEG_QUALITY = 0.8;
  const BURST_COUNT = 5;
  const BURST_DELAY_MS = 220;

  const forms = document.querySelectorAll("[data-camera-form]");
  if (!forms.length) {
    return;
  }

  forms.forEach((form) => {
    const video = form.querySelector("#camera-video");
    const canvas = form.querySelector("#camera-canvas");
    const preview = form.querySelector("#camera-preview");
    const hiddenInput = form.querySelector("#camera_image_samples");
    const status = form.querySelector("#camera-status");
    const startButton = form.querySelector("[data-camera-start]");
    const captureButton = form.querySelector("[data-camera-capture]");
    const retakeButton = form.querySelector("[data-camera-retake]");

    let stream = null;
    let capturedFrames = [];

    const setStatus = (message) => {
      if (status) {
        status.textContent = message;
      }
    };

    const startCamera = async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus("This browser does not support camera access.");
        return;
      }

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1024 },
            height: { ideal: 768 },
          },
          audio: false,
        });
        video.srcObject = stream;
        await video.play();
        setStatus("Camera is active. Center the face and press Capture.");
      } catch (error) {
        setStatus("Camera access was blocked. Allow permission and try again.");
      }
    };

    const delay = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

    const snapshot = () => {
      if (!video.videoWidth || !video.videoHeight) {
        return null;
      }

      const scale = Math.min(1, MAX_CAPTURE_WIDTH / video.videoWidth);
      canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
      canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    };

    const captureFrame = async () => {
      if (!video.videoWidth || !video.videoHeight) {
        setStatus("Start the camera before capturing.");
        return;
      }

      captureButton.disabled = true;
      retakeButton.disabled = true;
      capturedFrames = [];
      setStatus(`Capturing ${BURST_COUNT} samples from the camera...`);

      for (let index = 0; index < BURST_COUNT; index += 1) {
        const dataUrl = snapshot();
        if (dataUrl) {
          capturedFrames.push(dataUrl);
          preview.src = dataUrl;
        }
        if (index < BURST_COUNT - 1) {
          await delay(BURST_DELAY_MS);
        }
      }

      hiddenInput.value = JSON.stringify(capturedFrames);
      preview.classList.remove("hidden");
      video.classList.add("hidden");
      captureButton.disabled = false;
      retakeButton.disabled = false;
      setStatus(
        `Captured ${capturedFrames.length} sample(s). Submit to continue, or Retake if needed.`
      );
    };

    const retake = () => {
      capturedFrames = [];
      hiddenInput.value = "";
      preview.src = "";
      preview.classList.add("hidden");
      video.classList.remove("hidden");
      setStatus("Capture a new burst of images from the computer camera.");
    };

    startButton?.addEventListener("click", startCamera);
    captureButton?.addEventListener("click", captureFrame);
    retakeButton?.addEventListener("click", retake);

    form.addEventListener("submit", (event) => {
      if (!hiddenInput.value || hiddenInput.value === "[]") {
        event.preventDefault();
        setStatus("Capture camera samples before submitting.");
      }
    });

    window.addEventListener("beforeunload", () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    });

    startCamera();
  });
})();
