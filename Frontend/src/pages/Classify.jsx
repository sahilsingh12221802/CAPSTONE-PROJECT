import { useState } from 'react'
import ImageUpload from '../components/ImageUpload'

export default function Classify() {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)

  const handlePredict = () => {
    setLoading(true)
    setTimeout(() => {
      setPrediction({
        breed: 'Gir Cattle',
        confidence: '92%',
        description: 'This is an indigenous Indian cattle breed known for heat tolerance.'
      })
      setLoading(false)
    }, 1500)
  }

  const containerStyle = {
    padding: '40px 20px',
    maxWidth: '1200px',
    margin: '0 auto'
  }

  const headerStyle = {
    background: 'linear-gradient(135deg, #1a4d3a 0%, #2f7a60 100%)',
    color: 'white',
    padding: '40px 20px',
    borderRadius: '8px',
    marginBottom: '40px',
    textAlign: 'center'
  }

  const headerTitleStyle = {
    fontSize: '36px',
    fontWeight: 'bold',
    marginBottom: '10px'
  }

  const contentWrapperStyle = {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '30px',
    alignItems: 'start'
  }

  const sectionStyle = {
    backgroundColor: 'white',
    padding: '30px',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
  }

  const buttonStyle = {
    backgroundColor: '#1a4d3a',
    color: 'white',
    border: 'none',
    padding: '12px 30px',
    fontSize: '16px',
    fontWeight: '600',
    borderRadius: '4px',
    cursor: 'pointer',
    marginTop: '20px',
    width: '100%',
    transition: 'background-color 0.3s'
  }

  const resultStyle = {
    marginTop: '30px',
    padding: '20px',
    backgroundColor: '#f0f2f5',
    borderRadius: '8px',
    borderLeft: '4px solid #1a4d3a'
  }

  const resultTitleStyle = {
    fontSize: '20px',
    fontWeight: 'bold',
    color: '#1a4d3a',
    marginBottom: '15px'
  }

  return (
    <div style={containerStyle}>
      <div style={headerStyle}>
        <div style={headerTitleStyle}>Image Classification</div>
        <p style={{ opacity: '0.9' }}>Upload an image to classify the animal breed</p>
      </div>

      <div style={contentWrapperStyle}>
        <div style={sectionStyle}>
          <h3 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '20px', color: '#1a4d3a' }}>
            Upload Image
          </h3>
          <ImageUpload />
          <button
            style={buttonStyle}
            onClick={handlePredict}
            disabled={loading}
            onMouseEnter={(e) => !loading && (e.target.style.backgroundColor = '#2f7a60')}
            onMouseLeave={(e) => e.target.style.backgroundColor = '#1a4d3a'}
          >
            {loading ? 'Analyzing...' : 'Predict Breed'}
          </button>
        </div>

        <div style={sectionStyle}>
          <h3 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '20px', color: '#1a4d3a' }}>
            Results
          </h3>
          {prediction ? (
            <div style={resultStyle}>
              <div style={resultTitleStyle}>Prediction Result</div>
              <div style={{ marginBottom: '15px' }}>
                <strong style={{ color: '#1a4d3a' }}>Breed:</strong>
                <div style={{ fontSize: '18px', fontWeight: '600', color: '#2f7a60', marginTop: '5px' }}>
                  {prediction.breed}
                </div>
              </div>
              <div style={{ marginBottom: '15px' }}>
                <strong style={{ color: '#1a4d3a' }}>Confidence:</strong>
                <div style={{ fontSize: '18px', fontWeight: '600', color: '#2f7a60', marginTop: '5px' }}>
                  {prediction.confidence}
                </div>
              </div>
              <div>
                <strong style={{ color: '#1a4d3a' }}>Description:</strong>
                <div style={{ fontSize: '14px', color: '#666', marginTop: '5px' }}>
                  {prediction.description}
                </div>
              </div>
            </div>
          ) : (
            <div style={{ padding: '40px 20px', textAlign: 'center', color: '#999' }}>
              <p>No prediction yet</p>
              <p style={{ fontSize: '12px', marginTop: '10px' }}>Upload an image and click "Predict Breed"</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
