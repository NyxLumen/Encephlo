import React, { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Sphere, MeshDistortMaterial } from "@react-three/drei";

// --- 3D TUMOR MAPPER ---
// This component plots a glowing orb based on the diagnosis
const TumorVisualizer = ({ diagnosis }) => {
	// Default: No tumor, hidden inside the brain
	let position = [0, -10, 0];
	let color = "green";

	// Predictive Spatial Mapping based on your 4 classes
	if (diagnosis === "Glioma") {
		position = [1.5, 1, 0.5]; // Deep in the frontal/parietal lobe
		color = "red";
	} else if (diagnosis === "Meningioma") {
		position = [-2, 2, 1]; // Surface level
		color = "orange";
	} else if (diagnosis === "Pituitary") {
		position = [0, -1.5, 0.5]; // Base of the brain
		color = "purple";
	}

	return (
		<group>
			{/* Placeholder for the actual 3D Brain Model (.glb / .gltf) */}
			<Sphere
				args={[3, 32, 32]}
				position={[0, 0, 0]}
				wireframe
				opacity={0.1}
				transparent
			>
				<meshStandardMaterial color="cyan" />
			</Sphere>

			{/* The Tumor Heatmap Orb */}
			{diagnosis && diagnosis !== "No Tumor" && (
				<Sphere args={[0.5, 32, 32]} position={position}>
					<MeshDistortMaterial
						color={color}
						emissive={color}
						emissiveIntensity={2}
						distort={0.4}
						speed={2}
					/>
				</Sphere>
			)}
		</group>
	);
};

// --- MAIN DASHBOARD ---
const MriScanner = () => {
	const [file, setFile] = useState(null);
	const [preview, setPreview] = useState(null);
	const [loading, setLoading] = useState(false);
	const [result, setResult] = useState(null);

	const handleFileChange = (e) => {
		const selectedFile = e.target.files[0];
		if (selectedFile) {
			setFile(selectedFile);
			setPreview(URL.createObjectURL(selectedFile));
			setResult(null);
		}
	};

	const analyzeScan = async () => {
		if (!file) return;
		setLoading(true);

		const formData = new FormData();
		formData.append("file", file);

		try {
			const response = await fetch("http://localhost:8000/api/v1/analyze", {
				method: "POST",
				body: formData,
			});

			const data = await response.json();
			setResult(data);
		} catch (err) {
			console.error("Engine Connection Failed:", err);
		} finally {
			setLoading(false);
		}
	};

	return (
		<div
			className="dashboard-container"
			style={{
				display: "flex",
				gap: "20px",
				padding: "20px",
				backgroundColor: "#111",
				color: "white",
				minHeight: "100vh",
			}}
		>
			{/* LEFT PANEL: 2D Upload & Results */}
			<div
				className="upload-panel"
				style={{
					flex: 1,
					padding: "20px",
					border: "1px solid #333",
					borderRadius: "10px",
				}}
			>
				<h2>Encephlo Diagnostics</h2>

				<input
					type="file"
					onChange={handleFileChange}
					accept="image/*"
					style={{ marginBottom: "20px" }}
				/>

				{preview && (
					<img
						src={preview}
						alt="MRI"
						style={{
							maxWidth: "100%",
							height: "auto",
							borderRadius: "8px",
							marginBottom: "20px",
						}}
					/>
				)}

				<button
					onClick={analyzeScan}
					disabled={!file || loading}
					style={{
						padding: "10px 20px",
						backgroundColor: "#2563eb",
						color: "white",
						border: "none",
						borderRadius: "5px",
						cursor: "pointer",
						width: "100%",
					}}
				>
					{loading ? "Fusing 1,792 Dimensions..." : "Run Analysis"}
				</button>

				{result && (
					<div
						className="results-box"
						style={{
							marginTop: "20px",
							padding: "15px",
							backgroundColor: "#222",
							borderRadius: "8px",
						}}
					>
						<h3>
							Diagnosis:{" "}
							<span
								style={{
									color:
										result.diagnosis === "No Tumor" ? "#4ade80" : "#f87171",
								}}
							>
								{result.diagnosis}
							</span>
						</h3>
						<p>Confidence: {result.confidence}%</p>
						<p style={{ fontSize: "12px", color: "#888" }}>
							Inference: {result.inference_time_ms}ms
						</p>
					</div>
				)}
			</div>

			{/* RIGHT PANEL: 3D Visualization */}
			<div
				className="visualizer-panel"
				style={{
					flex: 2,
					border: "1px solid #333",
					borderRadius: "10px",
					overflow: "hidden",
					position: "relative",
				}}
			>
				<Canvas camera={{ position: [0, 0, 8] }}>
					<ambientLight intensity={0.5} />
					<pointLight position={[10, 10, 10]} />
					<OrbitControls
						enableZoom={true}
						autoRotate={!result}
						autoRotateSpeed={0.5}
					/>

					<TumorVisualizer diagnosis={result?.diagnosis} />
				</Canvas>
				<div
					style={{
						position: "absolute",
						bottom: "10px",
						left: "10px",
						color: "#888",
						pointerEvents: "none",
					}}
				>
					Interactive 3D Spatial Mapping
				</div>
			</div>
		</div>
	);
};

export default MriScanner;
