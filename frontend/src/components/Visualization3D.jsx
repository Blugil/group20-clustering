import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import './Visualization3D.css'

const Visualization3D = ({ data }) => {
  const mountRef = useRef(null)
  const sceneRef = useRef(null)
  const rendererRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const pointsRef = useRef(null)
  const raycasterRef = useRef(null)
  const mouseRef = useRef(new THREE.Vector2())
  const [hoveredId, setHoveredId] = useState(null)

  useEffect(() => {
    if (!mountRef.current || !data) return

    const width = mountRef.current.clientWidth
    const height = mountRef.current.clientHeight

    // Scene setup
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0a0a0a) // Dark background
    sceneRef.current = scene

    // Camera setup
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000)
    camera.position.z = 4
    cameraRef.current = camera

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(window.devicePixelRatio)
    mountRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controlsRef.current = controls

    // Raycaster for hover detection
    const raycaster = new THREE.Raycaster()
    raycaster.params.Points.threshold = 0.1 // Increase threshold for better hover detection
    raycasterRef.current = raycaster

    // Convert data to positions and colors arrays
    const dataEntries = Object.entries(data)
    const count = dataEntries.length
    const positions = new Float32Array(count * 3)
    const colors = new Float32Array(count * 3)
    const ids = []
    const clusters = []

    // ASU colors for clusters: Maroon, Gold, and variations
    const clusterColors = [
      new THREE.Color(0x8C1D40), // ASU Maroon
      new THREE.Color(0xFFC627), // ASU Gold
      new THREE.Color(0xB8860B), // Dark Goldenrod
      new THREE.Color(0xCD5C5C), // Indian Red
    ]

    dataEntries.forEach(([id, item], index) => {
      // Handle both old format [x, y, z] and new format {point: [x, y, z], cluster: n}
      const point = Array.isArray(item) ? item : item.point
      const cluster = Array.isArray(item) ? 0 : (item.cluster || 0)
      
      positions[index * 3 + 0] = point[0]
      positions[index * 3 + 1] = point[1]
      positions[index * 3 + 2] = point[2]
      
      // Set color based on cluster
      const color = clusterColors[cluster % clusterColors.length]
      colors[index * 3 + 0] = color.r
      colors[index * 3 + 1] = color.g
      colors[index * 3 + 2] = color.b
      
      ids.push(id)
      clusters.push(cluster)
    })

    // Create geometry and material
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))

    // Store IDs and clusters in geometry for hover detection
    geometry.userData.ids = ids
    geometry.userData.clusters = clusters

    const material = new THREE.PointsMaterial({
      size: 0.03,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.8,
      vertexColors: true // Use vertex colors for cluster coloring
    })

    const points = new THREE.Points(geometry, material)
    scene.add(points)
    pointsRef.current = points

    // Mouse move handler for hover
    const handleMouseMove = (event) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

      raycaster.setFromCamera(mouseRef.current, camera)
      const intersects = raycaster.intersectObject(points)

      if (intersects.length > 0) {
        const index = intersects[0].index
        const storedIds = geometry.userData.ids
        const storedClusters = geometry.userData.clusters
        const id = storedIds[index]
        const cluster = storedClusters[index]
        setHoveredId({ id, cluster })
      } else {
        setHoveredId(null)
      }
    }

    renderer.domElement.addEventListener('mousemove', handleMouseMove)

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    // Handle window resize
    const handleResize = () => {
      const newWidth = mountRef.current.clientWidth
      const newHeight = mountRef.current.clientHeight
      camera.aspect = newWidth / newHeight
      camera.updateProjectionMatrix()
      renderer.setSize(newWidth, newHeight)
    }
    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      renderer.domElement.removeEventListener('mousemove', handleMouseMove)
      if (mountRef.current && renderer.domElement.parentNode) {
        mountRef.current.removeChild(renderer.domElement)
      }
      renderer.dispose()
      geometry.dispose()
      material.dispose()
    }
  }, [data])

  return (
    <div className="visualization-container">
      <div ref={mountRef} className="visualization-canvas" />
      {hoveredId && (
        <div className="hover-tooltip">
          ID: {hoveredId.id}<br />
          Cluster: {hoveredId.cluster}
        </div>
      )}
    </div>
  )
}

export default Visualization3D

