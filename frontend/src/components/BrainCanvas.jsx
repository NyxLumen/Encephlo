import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";

const BrainModel = ({ heatmapUrl }) => {
	// 1. Load the file from the public folder
	// Change this to '/brain.gltf' if your file is a .gltf
	const { scene } = useGLTF("/brain.glb");

	// 2. Clone the scene so React doesn't complain about mutating cached objects
	const clonedScene = useMemo(() => scene.clone(), [scene]);

	// 3. The Material Override (The "Medical Glass" Look)
	useMemo(() => {
		clonedScene.traverse((child) => {
			if (child.isMesh) {
				// Strip the old pink/white color and apply our clean UI material
				child.material = new THREE.MeshStandardMaterial({
					color: "#ffffff",
					transparent: true,
					opacity: heatmapUrl ? 1 : 0.6, // Solid if there's a heatmap, ghostly if not
					roughness: 0.4,
					metalness: 0.1,
					wireframe: !heatmapUrl, // Shows a cool wireframe until the scan runs
				});
			}
		});
	}, [clonedScene, heatmapUrl]);

	// 4. Render it. You might need to tweak the scale=[1, 1, 1] depending on how big the Sketchfab model was.
	return <primitive object={clonedScene} scale={25} position={[0, 0, 0]} />;
};

export default function BrainCanvas({ heatmapUrl }) {
	return (
		<div
			style={{
				position: "relative",
				width: "100%",
				height: "70vh0",
				minHeight: "600px",
				backgroundColor: "#050505",
				borderRadius: "12px",
				border: "1px solid #222",
				overflow: "hidden",
			}}
		>
			<Canvas camera={{ position: [0, 0, 6], fov: 45 }}>
				{/* Cinematic UI Lighting */}
				<ambientLight intensity={0.5} />
				<directionalLight
					position={[10, 10, 10]}
					intensity={1}
					color="#ffffff"
				/>
				<directionalLight
					position={[-10, -10, -5]}
					intensity={0.5}
					color="#00ff88"
				/>

				<Suspense
					fallback={
						<mesh>
							<boxGeometry args={[1, 1, 1]} />
							<meshBasicMaterial color="red" wireframe />
						</mesh>
					}
				>
					<BrainModel heatmapUrl={heatmapUrl} />
				</Suspense>

				<OrbitControls
					enableZoom={true}
					autoRotate={!heatmapUrl}
					autoRotateSpeed={1.0}
					enablePan={false}
				/>
			</Canvas>
		</div>
	);
}
