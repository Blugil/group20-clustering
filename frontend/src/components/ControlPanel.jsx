import React, { useState } from 'react'
import './ControlPanel.css'

const ControlPanel = ({ onLoad, loading }) => {
  const [visualizationTechnique, setVisualizationTechnique] = useState('')
  const [embeddingAlgorithm, setEmbeddingAlgorithm] = useState('')
  const [clusteringAlgorithm, setClusteringAlgorithm] = useState('')

  const visualizationOptions = [
    { value: 'pca', label: 'PCA' },
    { value: 'tsne', label: 't-SNE' }
  ]

  const embeddingOptions = [
    { value: 'word2vec', label: 'Word2Vec' },
    { value: 'glove', label: 'GloVe' },
    { value: 'fasttext', label: 'FastText' },
    { value: 'bert', label: 'BERT' },
    { value: 'elmo', label: 'ELMo' }
  ]

  const clusteringOptions = [
    { value: 'kmeans', label: 'K-Means' },
    { value: 'dbscan', label: 'DBSCAN' },
    { value: 'hierarchical', label: 'Hierarchical' },
    { value: 'gmm', label: 'Gaussian Mixture Model' }
  ]

  const handleLoad = () => {
    if (!visualizationTechnique || !embeddingAlgorithm || !clusteringAlgorithm) {
      alert('Please select all options before loading data.')
      return
    }

    onLoad({
      visualizationTechnique,
      embeddingAlgorithm,
      clusteringAlgorithm
    })
  }

  return (
    <div className="control-panel">
      <h2>CONFIGURATION</h2>
      
      <div className="control-group">
        <label htmlFor="visualization">Visualization Technique</label>
        <select
          id="visualization"
          value={visualizationTechnique}
          onChange={(e) => setVisualizationTechnique(e.target.value)}
          disabled={loading}
        >
          <option value="">Select technique...</option>
          {visualizationOptions.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label htmlFor="embedding">Embedding Algorithm</label>
        <select
          id="embedding"
          value={embeddingAlgorithm}
          onChange={(e) => setEmbeddingAlgorithm(e.target.value)}
          disabled={loading}
        >
          <option value="">Select algorithm...</option>
          {embeddingOptions.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label htmlFor="clustering">Clustering Algorithm</label>
        <select
          id="clustering"
          value={clusteringAlgorithm}
          onChange={(e) => setClusteringAlgorithm(e.target.value)}
          disabled={loading}
        >
          <option value="">Select algorithm...</option>
          {clusteringOptions.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <button
        className="load-button"
        onClick={handleLoad}
        disabled={loading || !visualizationTechnique || !embeddingAlgorithm || !clusteringAlgorithm}
      >
        {loading ? 'Loading...' : 'Load'}
      </button>
    </div>
  )
}

export default ControlPanel

