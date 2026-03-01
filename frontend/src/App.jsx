import { useState } from "react";
import axios from "axios";
import BrainCanvas from "./components/BrainCanvas"; // <-- Import the 3D Canvas

function App() {
	const [selectedFile, setSelectedFile] = useState(null);
	const [previewUrl, setPreviewUrl] = useState(null);
	const [metrics, setMetrics] = useState(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);

	const handleFileChange = (e) => {
		const file = e.target.files[0];
		if (file) {
			setSelectedFile(file);
			setPreviewUrl(URL.createObjectURL(file));
			setMetrics(null);
			setError(null);
		}
	};

	const handleScan = async () => {
		if (!selectedFile) return;
		setLoading(true);
		setError(null);

		const formData = new FormData();
		formData.append("file", selectedFile);

		try {
			const response = await axios.post(
				"http://localhost:8000/api/v1/analyze",
				formData,
				{
					headers: { "Content-Type": "multipart/form-data" },
				},
			);
			setMetrics(response.data);
		} catch (err) {
			console.error(err);
			setError(
				err.response?.data?.detail || "Failed to connect to Neural Engine.",
			);
		} finally {
			setLoading(false);
		}
	};

	return (
		<div
			style={{
				fontFamily: "system-ui",
				padding: "40px 5%",
				maxWidth: "95%",
				margin: "0 auto",
				color: "#fff",
				backgroundColor: "#121212",
				minHeight: "100vh",
			}}
		>
			<h1>
				🧠 Encephlo 3.0{" "}
				<span style={{ fontSize: "0.5em", color: "#888" }}>
					Volumetric Diagnostics
				</span>
			</h1>

			<div
				style={{
					display: "grid",
					gridTemplateColumns: "1fr 2fr",
					gap: "40px",
					marginTop: "30px",
				}}
			>
				{/* LEFT COLUMN: Controls & Upload */}
				<div
					style={{
						padding: "20px",
						backgroundColor: "#1e1e1e",
						borderRadius: "12px",
						border: "1px solid #333",
					}}
				>
					<h3>Data Input</h3>
					<input
						type="file"
						accept="image/*"
						onChange={handleFileChange}
						style={{ marginBottom: "20px" }}
					/>

					{previewUrl && (
						<div style={{ marginBottom: "20px" }}>
							<img
								src={previewUrl}
								alt="MRI Preview"
								style={{
									width: "100%",
									borderRadius: "8px",
									border: "1px solid #444",
								}}
							/>
						</div>
					)}

					<button
						onClick={handleScan}
						disabled={!selectedFile || loading}
						style={{
							width: "100%",
							padding: "15px",
							fontSize: "16px",
							cursor: "pointer",
							backgroundColor: "#00ff88",
							color: "#000",
							border: "none",
							borderRadius: "8px",
							fontWeight: "bold",
						}}
					>
						{loading ? "Running Multimodal Fusion..." : "Initialize Scan"}
					</button>

					{error && (
						<div style={{ marginTop: "15px", color: "#ff4444" }}>
							⚠️ {error}
						</div>
					)}

					{metrics && (
						<div
							style={{
								marginTop: "20px",
								padding: "15px",
								backgroundColor: "#0a0a0a",
								borderRadius: "8px",
								border: "1px solid #333",
							}}
						>
							<h4 style={{ margin: "0 0 10px 0", color: "#888" }}>
								Ensemble Consensus
							</h4>
							<h2
								style={{
									margin: "0 0 5px 0",
									color:
										metrics.diagnosis === "No Tumor" ? "#00ff88" : "#ff4444",
								}}
							>
								{metrics.diagnosis}
							</h2>
							<p style={{ margin: 0 }}>
								Confidence: <strong>{metrics.confidence}%</strong>
							</p>
							<p style={{ margin: 0, fontSize: "0.8em", color: "#888" }}>
								Inference: {metrics.inference_time_ms}ms
							</p>
						</div>
					)}
				</div>

				{/* RIGHT COLUMN: 3D XAI Interface */}
				<div style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
					<h3 style={{ marginTop: 0 }}>Spatial Attention (ScoreCAM)</h3>
					{/* We pass the heatmap_url to the canvas if it exists */}
					<BrainCanvas heatmapUrl={metrics?.heatmap_url} />
					<p
						style={{
							fontSize: "0.8em",
							color: "#666",
							textAlign: "center",
							marginTop: "10px",
						}}
					>
						Left Click + Drag to Rotate | Scroll to Zoom
					</p>
				</div>
			</div>
		</div>
	);
}

export default App;
