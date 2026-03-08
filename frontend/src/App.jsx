import React, { useState } from "react";
import BrainCanvas from "./components/BrainCanvas";
import "./App.css";

export default function MainDashboard() {
	const [file, setFile] = useState(null);
	const [preview, setPreview] = useState(null);
	const [loading, setLoading] = useState(false);
	const [result, setResult] = useState(null);
	const [error, setError] = useState(null);

	const getHeatmapTexture = (diagnosis) => {
		switch (diagnosis) {
			case "Glioma":
				return "/heatmaps/glioma_glow.jpg";
			case "Meningioma":
				return "/heatmaps/meningioma_glow.jpg";
			case "Pituitary":
				return "/heatmaps/pituitary_glow.jpg";
			default:
				return "/dummy_map.jpg";
		}
	};

	const handleFileChange = (e) => {
		const selectedFile = e.target.files[0];
		if (selectedFile) {
			setFile(selectedFile);
			setPreview(URL.createObjectURL(selectedFile));
			setResult(null);
			setError(null);
		}
	};

	const analyzeScan = async () => {
		if (!file) return;
		setLoading(true);
		setError(null);

		const formData = new FormData();
		formData.append("file", file);

		try {
			const response = await fetch("http://localhost:8000/api/v1/analyze", {
				method: "POST",
				body: formData,
			});

			if (!response.ok) throw new Error("Backend connection failed.");

			const data = await response.json();
			setResult(data);
		} catch (err) {
			setError(err.message || "Failed to connect to the Neural Engine.");
		} finally {
			setLoading(false);
		}
	};

	return (
		<div className="dashboard-container">
			<header className="dashboard-header">
				<h1>Encephlo 3.0 // Neural Fusion</h1>
				<p>1,792-Dimensional Hybrid SVM Architecture</p>
			</header>

			<div className="dashboard-layout">
				{/* LEFT COLUMN: Controls & Results */}
				{/* LEFT COLUMN: Controls & Results */}
				<div className="control-panel">
					<div className="upload-card">
						<h2>Input MRI Scan</h2>

						<div className="upload-dropzone">
							<input
								type="file"
								accept="image/*"
								onChange={handleFileChange}
								id="mri-upload"
							/>
							<label htmlFor="mri-upload">
								{/* 1. Show ONLY the original unedited MRI here */}
								{preview ? (
									<img
										src={preview}
										alt="MRI Preview"
										className="preview-image"
									/>
								) : (
									<div className="upload-placeholder">
										<span>Click or Drag MRI Here</span>
									</div>
								)}
							</label>
						</div>

						<button
							onClick={analyzeScan}
							disabled={!file || loading}
							className={`action-btn ${loading || !file ? "disabled" : "active"}`}
						>
							{loading ? "Fusing Feature Vectors..." : "Run SVM Inference"}
						</button>
					</div>

					{error && <div className="error-banner">{error}</div>}

					{result && (
						<>
							{/* Diagnostic Metrics */}
							<div className="results-card">
								<h3>Diagnostic Output</h3>

								<div className="diagnosis-section">
									<span className="label">Classification</span>
									<span
										className={`diagnosis-text ${result.diagnosis === "No Tumor" ? "safe" : "danger"}`}
									>
										{result.diagnosis}
									</span>
								</div>

								<div className="metrics-grid">
									<div className="metric-box">
										<span className="label">Confidence</span>
										<span className="value confidence">
											{result.confidence}%
										</span>
									</div>
									<div className="metric-box">
										<span className="label">Latency</span>
										<span className="value latency">
											{result.inference_time_ms}ms
										</span>
									</div>
								</div>
							</div>

							{/* 2. NEW: Dedicated Score-CAM Heatmap Panel */}
							{result.heatmap_url && (
								<div
									className="results-card"
									style={{
										marginTop: "1rem",
										border: "1px solid #3b82f6",
										boxShadow: "0 0 15px rgba(59, 130, 246, 0.1)",
									}}
								>
									<h3 style={{ color: "#60a5fa" }}>CNN Spatial Activation</h3>
									<div style={{ textAlign: "center", marginTop: "1rem" }}>
										<img
											src={result.heatmap_url}
											alt="Score-CAM"
											style={{
												width: "100%",
												borderRadius: "8px",
												border: "1px solid #1f2937",
											}}
										/>
										<p
											style={{
												fontSize: "11px",
												color: "#9ca3af",
												marginTop: "8px",
											}}
										>
											Thermal signature indicating DenseNet pixel attention
										</p>
									</div>
								</div>
							)}
						</>
					)}
				</div>

				{/* RIGHT COLUMN: The 3D Holographic Brain */}
				<div className="visualization-panel">
					<div className="canvas-wrapper">
						<div className="tech-overlay">
							<p>OBJ: HUMAN_BRAIN_01</p>
							<p>MAT: ADDITIVE_BLEND_HOLO</p>
							<p>
								MAP: {result ? result.diagnosis.toUpperCase() : "AWAITING_DATA"}
							</p>
						</div>

						<BrainCanvas
							heatmapUrl={result ? getHeatmapTexture(result.diagnosis) : null}
						/>
					</div>
				</div>
			</div>
		</div>
	);
}
