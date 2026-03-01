import { useState } from "react";
import axios from "axios";

function App() {
	const [selectedFile, setSelectedFile] = useState(null);
	const [previewUrl, setPreviewUrl] = useState(null);
	const [metrics, setMetrics] = useState(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);

	// Handle file selection and generate a local preview
	const handleFileChange = (e) => {
		const file = e.target.files[0];
		if (file) {
			setSelectedFile(file);
			setPreviewUrl(URL.createObjectURL(file));
			setMetrics(null);
			setError(null);
		}
	};

	// The actual API Call to your FastAPI Backend
	const handleScan = async () => {
		if (!selectedFile) return;

		setLoading(true);
		setError(null);

		const formData = new FormData();
		formData.append("file", selectedFile);

		try {
			// Hitting the endpoint we built in main.py
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
				padding: "40px",
				maxWidth: "800px",
				margin: "0 auto",
				color: "#fff",
				backgroundColor: "#121212",
				minHeight: "100vh",
			}}
		>
			<h1>
				🧠 Encephlo 3.0 |{" "}
				<span style={{ color: "#00ff88" }}>Tracer Bullet</span>
			</h1>
			<p>Upload an MRI scan to test the FastAPI backend pipeline.</p>

			<div
				style={{
					margin: "20px 0",
					padding: "20px",
					border: "1px dashed #444",
					borderRadius: "8px",
				}}
			>
				<input type="file" accept="image/*" onChange={handleFileChange} />

				{previewUrl && (
					<div style={{ marginTop: "20px" }}>
						<img
							src={previewUrl}
							alt="MRI Preview"
							style={{ maxWidth: "200px", borderRadius: "8px" }}
						/>
					</div>
				)}
			</div>

			<button
				onClick={handleScan}
				disabled={!selectedFile || loading}
				style={{
					padding: "10px 20px",
					fontSize: "16px",
					cursor: "pointer",
					backgroundColor: "#00ff88",
					color: "#000",
					border: "none",
					borderRadius: "4px",
					fontWeight: "bold",
				}}
			>
				{loading ? "Analyzing..." : "Run Neural Diagnostic"}
			</button>

			{error && (
				<div
					style={{
						marginTop: "20px",
						padding: "15px",
						backgroundColor: "#ff444422",
						border: "1px solid #ff4444",
						color: "#ff4444",
						borderRadius: "4px",
					}}
				>
					⚠️ {error}
				</div>
			)}

			{metrics && (
				<div
					style={{
						marginTop: "30px",
						padding: "20px",
						backgroundColor: "#1e1e1e",
						borderRadius: "8px",
						border: "1px solid #333",
					}}
				>
					<h3>✅ Backend Response Received</h3>
					<pre style={{ color: "#00ff88", overflowX: "auto" }}>
						{JSON.stringify(metrics, null, 2)}
					</pre>
				</div>
			)}
		</div>
	);
}

export default App;
