import { useState } from 'react'
import ImagePreview from './ImagePreview'

export default function ImageUpload() {
  const [image, setImage] = useState(null)

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImage(file)
    }
  }

  const containerStyle = {
    backgroundColor: 'white',
    borderRadius: '8px',
    padding: '25px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
  }

  const titleStyle = {
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#1a4d3a',
    marginBottom: '20px'
  }

  const uploadAreaStyle = {
    border: '2px dashed #ccc',
    borderRadius: '8px',
    padding: '40px 20px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 0.3s',
    backgroundColor: '#f9faf8'
  }

  const uploadTextStyle = {
    color: '#999',
    fontSize: '14px'
  }

  return (
    <div style={containerStyle}>
      <div style={titleStyle}>Select Animal Image</div>

      <label
        style={uploadAreaStyle}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = '#1a4d3a'
          e.currentTarget.style.backgroundColor = '#f0f2f5'
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.borderColor = '#ccc'
          e.currentTarget.style.backgroundColor = '#f9faf8'
        }}
      >
        <input
          type="file"
          accept="image/png, image/jpeg, image/jpg"
          onChange={handleImageChange}
          style={{ display: 'none' }}
        />
        <div style={{ fontSize: '28px', marginBottom: '10px' }}>📷</div>
        <div style={uploadTextStyle}>Click to upload PNG or JPG</div>
        <div style={{ fontSize: '12px', color: '#ccc', marginTop: '8px' }}>Maximum 5MB</div>
      </label>

      {image && <ImagePreview image={image} />}
    </div>
  )
}
