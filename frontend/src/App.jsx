import React, { useState } from 'react'
import Visualization3D from './components/Visualization3D'
import ControlPanel from './components/ControlPanel'
import './App.css'

function App() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [params, setParams] = useState({
    visualizationTechnique: '',
    embeddingAlgorithm: '',
    clusteringAlgorithm: ''
  })

  const handleLoad = async (newParams) => {
    setParams(newParams)
    setLoading(true)
    
    try {
      // Construct filename from parameters
      const filename = `${newParams.visualizationTechnique}_${newParams.embeddingAlgorithm}_${newParams.clusteringAlgorithm}.json`
      const response = await fetch(`/data/${filename}`)
      
      if (!response.ok) {
        throw new Error(`Failed to load data file: ${filename}`)
      }
      
      const jsonData = await response.json()
      setData(jsonData)
    } catch (error) {
      console.error('Error loading data:', error)
      alert(`Failed to load data. Please check that the file exists for the selected combination.`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Embedding Visualizer</h1>
        <div className="header-info">
          {data && (
            <span>
              Points: {Object.keys(data).length} | Dimension: 3
            </span>
          )}
        </div>
      </header>
      
      <div className="app-content">
        <div className="left-panel">
          <div className="panel-section">
            <h2>DATA</h2>
            <p>Select parameters in the right panel and click Load to visualize data.</p>
          </div>
        </div>
        
        <div className="center-panel">
          {loading ? (
            <div className="loading">Loading data...</div>
          ) : (
            <Visualization3D data={data} />
          )}
        </div>
        
        <div className="right-panel">
          <ControlPanel onLoad={handleLoad} loading={loading} />
        </div>
      </div>
    </div>
  )
}

export default App

