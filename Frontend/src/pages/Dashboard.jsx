export default function Dashboard() {
  const breeds = [
    { id: 1, name: 'Gir', type: 'Cattle', description: 'Indigenous Indian cattle breed' },
    { id: 2, name: 'Holstein', type: 'Cattle', description: 'High-yield dairy breed' },
    { id: 3, name: 'Jersey', type: 'Cattle', description: 'Premium quality milk' },
    { id: 4, name: 'Sahiwal', type: 'Cattle', description: 'Heat tolerant breed' },
    { id: 5, name: 'Murrah', type: 'Buffalo', description: 'Elite buffalo breed' },
    { id: 6, name: 'Jaffrabadi', type: 'Buffalo', description: 'Heavy productive breed' }
  ]

  const containerStyle = {
    padding: '40px 20px',
    maxWidth: '1400px',
    margin: '0 auto'
  }

  const headerStyle = {
    background: 'linear-gradient(135deg, #1a4d3a 0%, #2f7a60 100%)',
    color: 'white',
    padding: '50px 20px',
    borderRadius: '8px',
    marginBottom: '40px',
    textAlign: 'center'
  }

  const headerTitleStyle = {
    fontSize: '40px',
    fontWeight: 'bold',
    marginBottom: '15px'
  }

  const headerSubtitleStyle = {
    fontSize: '16px',
    opacity: '0.9'
  }

  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '20px',
    marginBottom: '40px'
  }

  const cardStyle = {
    backgroundColor: 'white',
    borderRadius: '8px',
    padding: '25px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
    transition: 'transform 0.3s, box-shadow 0.3s'
  }

  const cardTitleStyle = {
    fontSize: '20px',
    fontWeight: 'bold',
    color: '#1a4d3a',
    marginBottom: '10px'
  }

  const cardTextStyle = {
    fontSize: '14px',
    color: '#666',
    marginBottom: '8px'
  }

  return (
    <div style={containerStyle}>
      <div style={headerStyle}>
        <div style={headerTitleStyle}>Animal Classification Dashboard</div>
        <div style={headerSubtitleStyle}>Identify cattle and buffalo breeds using AI</div>
      </div>

      <div style={{ marginBottom: '30px' }}>
        <h2 style={{ fontSize: '28px', fontWeight: 'bold', marginBottom: '20px', color: '#1a4d3a' }}>
          Available Breeds
        </h2>
        <div style={gridStyle}>
          {breeds.map((breed) => (
            <div
              key={breed.id}
              style={cardStyle}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-5px)'
                e.currentTarget.style.boxShadow = '0 8px 16px rgba(0, 0, 0, 0.15)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)'
              }}
            >
              <div style={cardTitleStyle}>{breed.name}</div>
              <div style={cardTextStyle}><strong>Type:</strong> {breed.type}</div>
              <div style={cardTextStyle}>{breed.description}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ background: 'white', padding: '30px', borderRadius: '8px', textAlign: 'center', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)' }}>
        <h3 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '15px', color: '#1a4d3a' }}>Ready to Classify?</h3>
        <p style={{ color: '#666', marginBottom: '20px' }}>Go to the Classify page to upload an image and get predictions</p>
        <a 
          href="/classify"
          style={{
            display: 'inline-block',
            backgroundColor: '#1a4d3a',
            color: 'white',
            padding: '12px 30px',
            borderRadius: '4px',
            textDecoration: 'none',
            fontSize: '16px',
            fontWeight: '500',
            transition: 'background-color 0.3s'
          }}
          onMouseEnter={(e) => e.target.style.backgroundColor = '#2f7a60'}
          onMouseLeave={(e) => e.target.style.backgroundColor = '#1a4d3a'}
        >
          Go to Classify
        </a>
      </div>
    </div>
  )
}
