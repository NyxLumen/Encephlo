import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF, useTexture } from "@react-three/drei";
import * as THREE from "three";

const BrainModel = ({ heatmapUrl }) => {
	const { scene } = useGLTF("/brain.glb");

	// We need TWO clones now. One for the glass, one for the hologram.
	const baseScene = useMemo(() => scene.clone(), [scene]);
	const overlayScene = useMemo(() => scene.clone(), [scene]);

	const textureToLoad = heatmapUrl || "/dummy_map.jpg";
	const rawTexture = useTexture(textureToLoad);

	const heatmapTexture = useMemo(() => {
		const tex = rawTexture.clone();
		tex.flipY = false;
		tex.colorSpace = THREE.SRGBColorSpace;
		tex.needsUpdate = true;
		return tex;
	}, [rawTexture]);

	// 1. The Glass Base Layer
	useMemo(() => {
		baseScene.traverse((child) => {
			if (child.isMesh) {
				child.material = new THREE.MeshStandardMaterial({
					color: "#ffffff",
					transparent: true,
					opacity: 0.15, // Very faint ghostly glass
					roughness: 0.1,
					metalness: 0.5,
				});
			}
		});
	}, [baseScene]);

	// 2. The Holographic Heatmap Layer
	useMemo(() => {
		overlayScene.traverse((child) => {
			if (child.isMesh) {
				child.material = new THREE.MeshStandardMaterial({
					map: heatmapTexture,
					emissiveMap: heatmapTexture,
					emissiveIntensity: 2.0, // Makes the colors pop
					transparent: true,
					blending: THREE.AdditiveBlending, // THE MAGIC: Black pixels become invisible!
					depthWrite: false, // Stops the two meshes from glitching into each other
				});
			}
		});
	}, [overlayScene, heatmapTexture]);

	return (
		<group scale={25}>
			{/* Render the glass brain */}
			<primitive object={baseScene} />
			{/* Render the heatmap overlay slightly larger (1.01x) so it floats on top */}
			<primitive object={overlayScene} scale={1.01} />
		</group>
	);
};

export default function BrainCanvas({ heatmapUrl }) {
	return (
		<div
			style={{
				position: "relative",
				width: "100%",
				height: "70vh",
				minHeight: "600px",
				backgroundColor: "#050505",
				borderRadius: "12px",
				border: "1px solid #222",
				overflow: "hidden",
			}}
		>
			<Canvas camera={{ position: [0, 0, 6], fov: 45 }}>
				<ambientLight intensity={0.2} />
				<directionalLight
					position={[10, 10, 10]}
					intensity={0.5}
					color="#ffffff"
				/>

				<Suspense fallback={null}>
					<BrainModel heatmapUrl={heatmapUrl} />
				</Suspense>

				<OrbitControls
					enableZoom={true}
					autoRotate={true}
					autoRotateSpeed={1}
					enablePan={false}
				/>
			</Canvas>
		</div>
	);
}
