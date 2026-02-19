export default function ImagePreview({ image }) {
  const containerStyle = {
    marginTop: '20px',
    paddingTop: '20px',
    borderTop: '1px solid #eee'
  }

  const labelStyle = {
    fontSize: '12px',
    color: '#999',
    marginBottom: '10px',
    display: 'block'
  }

  const previewWrapperStyle = {
    border: '1px solid #ddd',
    borderRadius: '8px',
    overflow: 'hidden',
    marginBottom: '15px'
  }

  const imageStyle = {
    width: '100%',
    height: '200px',
    objectFit: 'contain',
    backgroundColor: '#f9faf8'
  }

  const infoStyle = {
    display: 'grid',
    gap: '10px'
  }

  const infoItemStyle = {
    fontSize: '13px',
    color: '#666',
    wordBreak: 'break-word'
  }

  return (
    <div style={containerStyle}>
      <label style={labelStyle}>Preview</label>

      <div style={previewWrapperStyle}>
        <img
          src={URL.createObjectURL(image)}
          alt="selected"
          style={imageStyle}
        />
      </div>

      <div style={infoStyle}>
        <div style={infoItemStyle}>
          <strong>File:</strong> {image.name}
        </div>
        <div style={infoItemStyle}>
          <strong>Size:</strong> {(image.size / 1024).toFixed(2)} KB
        </div>
      </div>
    </div>
  )
}
