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
									<span className="value confidence">{result.confidence}%</span>
								</div>
								<div className="metric-box">
									<span className="label">Latency</span>
									<span className="value latency">
										{result.inference_time_ms}ms
									</span>
								</div>
							</div>
						</div>
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
