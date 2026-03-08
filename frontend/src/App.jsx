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
		<div className="shadcn-container">
			{/* Navbar */}
			<nav className="shadcn-nav">
				<div className="nav-brand">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						strokeWidth="2"
						strokeLinecap="round"
						strokeLinejoin="round"
						className="brand-icon"
					>
						<path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
					</svg>
					<span className="font-semibold tracking-tight">Encephlo</span>
					<span className="badge">Beta</span>
				</div>
			</nav>

			<main className="shadcn-grid">
				{/* LEFT COLUMN: Controls */}
				<div className="flex-col gap-6">
					<div className="card">
						<div className="card-header">
							<h3 className="card-title">Diagnostic Interface</h3>
							<p className="card-description">
								Upload an MRI scan to generate a hybrid SVM & ViT analysis.
							</p>
						</div>

						<div className="card-content">
							<div className="upload-area">
								<input
									type="file"
									accept="image/*"
									onChange={handleFileChange}
									id="mri-upload"
									className="hidden-input"
								/>
								<label htmlFor="mri-upload" className="upload-label">
									{preview ? (
										<img src={preview} alt="MRI Scan" className="preview-img" />
									) : (
										<div className="upload-empty-state">
											<svg
												xmlns="http://www.w3.org/2000/svg"
												width="24"
												height="24"
												viewBox="0 0 24 24"
												fill="none"
												stroke="currentColor"
												strokeWidth="2"
												strokeLinecap="round"
												strokeLinejoin="round"
												className="text-muted"
											>
												<rect
													width="18"
													height="18"
													x="3"
													y="3"
													rx="2"
													ry="2"
												/>
												<circle cx="9" cy="9" r="2" />
												<path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
											</svg>
											<span className="text-sm font-medium mt-2">
												Click to upload
											</span>
											<span className="text-xs text-muted">
												JPEG, PNG or DICOM
											</span>
										</div>
									)}
								</label>
							</div>

							<button
								onClick={analyzeScan}
								disabled={!file || loading}
								className={`btn-primary w-full mt-4 ${loading || !file ? "opacity-50 cursor-not-allowed" : ""}`}
							>
								{loading ? (
									<span className="flex-center gap-2">
										<svg
											className="spinner"
											xmlns="http://www.w3.org/2000/svg"
											width="24"
											height="24"
											viewBox="0 0 24 24"
											fill="none"
											stroke="currentColor"
											strokeWidth="2"
											strokeLinecap="round"
											strokeLinejoin="round"
										>
											<path d="M21 12a9 9 0 1 1-6.219-8.56" />
										</svg>
										Processing...
									</span>
								) : (
									"Run Analysis"
								)}
							</button>
						</div>
					</div>

					{error && (
						<div className="alert alert-destructive">
							<span className="font-medium">Error:</span> {error}
						</div>
					)}

					{result && (
						<>
							{/* Results Card */}
							<div className="card mt-6">
								<div className="card-header">
									<h3 className="card-title">Analysis Results</h3>
								</div>
								<div className="card-content">
									<div className="flex justify-between items-baseline mb-4">
										<span className="text-sm font-medium text-muted">
											Classification
										</span>
										<span
											className={`text-2xl font-bold tracking-tight ${result.diagnosis === "No Tumor" ? "text-success" : "text-destructive"}`}
										>
											{result.diagnosis}
										</span>
									</div>

									<div className="separator"></div>

									<div className="grid-2-col pt-4">
										<div>
											<span className="text-xs text-muted block mb-1">
												Confidence Score
											</span>
											<span className="text-lg font-semibold">
												{result.confidence}%
											</span>
										</div>
										<div>
											<span className="text-xs text-muted block mb-1">
												Inference Latency
											</span>
											<span className="text-lg font-semibold">
												{result.inference_time_ms}ms
											</span>
										</div>
									</div>
								</div>
							</div>

							{/* Score-CAM / ViT Card */}
							{result.heatmap_url && (
								<div className="card mt-6">
									<div className="card-header pb-2">
										<h3 className="card-title text-sm">
											Spatial Attention Map
										</h3>
									</div>
									<div className="card-content">
										<img
											src={result.heatmap_url}
											alt="Attention Map"
											className="rounded-md border object-cover w-full"
										/>
									</div>
								</div>
							)}
						</>
					)}
				</div>

				{/* RIGHT COLUMN: 3D Visualization */}
				<div className="canvas-wrapper">
					<BrainCanvas
						heatmapUrl={result ? getHeatmapTexture(result.diagnosis) : null}
					/>

					<div className="floating-badge">
						<span className="flex items-center gap-1.5">
							<span className="status-indicator"></span>
							Live 3D Mapping
						</span>
					</div>
				</div>
			</main>
		</div>
	);
}
