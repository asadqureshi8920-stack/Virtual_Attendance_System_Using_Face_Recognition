(() => {
  const MAX_CAPTURE_WIDTH = 1024;
  const JPEG_QUALITY = 0.8;
  const BURST_COUNT = 5;
  const PREVIEW_BURST_COUNT = 3;
  const BURST_DELAY_MS = 220;
  const PREVIEW_INTERVAL_MS = 1100;
  const ENROLLMENT_STABLE_DELAY_MS = 700;

  const form = document.querySelector("[data-auto-attendance-form]");
  if (!form) {
    return;
  }

  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || "";
  const requestedSessionId = form.dataset.sessionId || "";
  const video = form.querySelector("#camera-video");
  const canvas = form.querySelector("#camera-canvas");
  const preview = form.querySelector("#camera-preview");
  const status = form.querySelector("#camera-status");
  const enrollmentInput = form.querySelector("#enrollment_no");
  const liveLabel = document.querySelector("#live-label");
  const liveConfidence = document.querySelector("#live-confidence");
  const liveMessage = document.querySelector("#live-message");
  const liveSession = document.querySelector("#live-session");
  const confidenceTrack = document.querySelector(".confidence-track span");

  let stream = null;
  let previewTimer = null;
  let restartTimer = null;
  let enrollmentDelayTimer = null;
  let isProcessing = false;
  let readyDetections = 0;
  let completed = false;

  const setStatus = (message) => {
    if (status) {
      status.textContent = message;
    }
  };

  const updateLiveCard = (data) => {
    if (liveLabel) {
      liveLabel.textContent = data.label || "Unknown";
    }
    if (liveConfidence) {
      const confidenceValue = Number(data.confidence_percent || 0).toFixed(2);
      liveConfidence.textContent = `${confidenceValue}%`;
    }
    if (confidenceTrack) {
      const confidenceValue = Math.max(0, Math.min(100, Number(data.confidence_percent || 0)));
      confidenceTrack.style.width = `${confidenceValue}%`;
      confidenceTrack.classList.toggle("is-live", confidenceValue > 0);
    }
    if (liveMessage) {
      liveMessage.textContent = data.message || "Waiting for recognition.";
    }
    if (liveSession) {
      const sessionText = data.session_id ? `${data.class_name || "Class"} | ${data.session_id}` : "Waiting for active session.";
      liveSession.textContent = sessionText;
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
      setStatus("Camera is active. Type the enrollment number and keep the face centered.");
    } catch (_error) {
      setStatus("Camera access was blocked. Allow permission and reload the page.");
    }
  };

  const postJson = async (url, payload) => {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrfToken,
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    return { ok: response.ok, data };
  };

  const collectSamples = async () => {
    return collectBurst(BURST_COUNT);
  };

  const collectPreviewSamples = async () => {
    return collectBurst(PREVIEW_BURST_COUNT);
  };

  const collectBurst = async (count) => {
    const samples = [];
    for (let index = 0; index < count; index += 1) {
      const dataUrl = snapshot();
      if (dataUrl) {
        samples.push(dataUrl);
        preview.src = dataUrl;
      }
      if (index < count - 1) {
        await delay(BURST_DELAY_MS);
      }
    }
    return samples;
  };

  const resetForRetry = () => {
    completed = false;
    readyDetections = 0;
    preview.classList.add("hidden");
    video.classList.remove("hidden");
    if (previewTimer) {
      window.clearInterval(previewTimer);
    }
    startPreviewLoop();
  };

  const submitAttendance = async () => {
    if (completed || isProcessing) {
      return;
    }
    if (!enrollmentInput.value.trim()) {
      return;
    }
    if (!video.videoWidth || !video.videoHeight) {
      setStatus("Waiting for the camera to become ready.");
      return;
    }

    isProcessing = true;
    setStatus("Capturing face samples automatically...");
    const samples = await collectSamples();
    preview.classList.remove("hidden");
    video.classList.add("hidden");
    setStatus("Submitting attendance...");

    const { ok, data } = await postJson("/attendance/api/check-in", {
      enrollment_no: enrollmentInput.value.trim(),
      session_id: requestedSessionId,
      camera_image_samples: samples,
    });
    updateLiveCard(data);
    setStatus(data.message || "Attendance processed.");
    completed = ok && ["Present", "Duplicate"].includes(data.status);
    isProcessing = false;

    if (!completed) {
      restartTimer = window.setTimeout(() => {
        resetForRetry();
      }, 2200);
    }
  };

  const runPreview = async () => {
    if (isProcessing || completed) {
      return;
    }
    if (!enrollmentInput.value.trim()) {
      return;
    }
    isProcessing = true;
    const previewSamples = await collectPreviewSamples();
    if (!previewSamples.length) {
      setStatus("Waiting for the camera to become ready.");
      isProcessing = false;
      return;
    }

    const { ok, data } = await postJson("/attendance/api/identify", {
      enrollment_no: enrollmentInput.value.trim(),
      session_id: requestedSessionId,
      camera_image_samples: previewSamples,
    });
    updateLiveCard(data);
    setStatus(data.message || "Recognition preview updated.");
    isProcessing = false;

    if (ok && data.face_detected) {
      readyDetections += 1;
    } else {
      readyDetections = 0;
    }

    if (readyDetections >= 2) {
      if (previewTimer) {
        window.clearInterval(previewTimer);
      }
      submitAttendance();
    }
  };

  const startPreviewLoop = () => {
    preview.classList.add("hidden");
    video.classList.remove("hidden");
    if (!enrollmentInput.value.trim()) {
      setStatus("Type the enrollment number to begin automatic attendance.");
      return;
    }
    setStatus("Enrollment number detected. Hold still while face recognition starts.");
    previewTimer = window.setInterval(runPreview, PREVIEW_INTERVAL_MS);
    runPreview();
  };

  const schedulePreviewLoop = () => {
    if (previewTimer) {
      window.clearInterval(previewTimer);
    }
    if (restartTimer) {
      window.clearTimeout(restartTimer);
    }
    if (enrollmentDelayTimer) {
      window.clearTimeout(enrollmentDelayTimer);
    }
    completed = false;
    readyDetections = 0;
    updateLiveCard({
      label: "Unknown",
      confidence_percent: 0,
      message: "Preparing recognition.",
      session_id: "",
    });

    enrollmentDelayTimer = window.setTimeout(startPreviewLoop, ENROLLMENT_STABLE_DELAY_MS);
  };

  enrollmentInput?.addEventListener("input", schedulePreviewLoop);
  form.addEventListener("submit", (event) => event.preventDefault());

  window.addEventListener("beforeunload", () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
  });

  startCamera();
  enrollmentInput?.focus();
})();
