import { Link } from 'react-router-dom'

export default function Navbar() {
  const navStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#1a4d3a',
    padding: '15px 40px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)'
  }

  const titleStyle = {
    fontSize: '24px',
    fontWeight: 'bold',
    color: 'white'
  }

  const navLinksStyle = {
    display: 'flex',
    gap: '30px',
    alignItems: 'center'
  }

  const linkStyle = {
    color: 'white',
    textDecoration: 'none',
    fontSize: '16px',
    fontWeight: '500',
    padding: '8px 16px',
    borderRadius: '4px',
    transition: 'all 0.3s ease'
  }

  return (
    <nav style={navStyle}>
      <div style={titleStyle}>Animal Classification</div>
      <div style={navLinksStyle}>
        <Link
          to="/"
          style={linkStyle}
          onMouseEnter={(e) => {
            e.target.backgroundColor = '#2f7a60'
            e.target.transform = 'scale(1.05)'
          }}
          onMouseLeave={(e) => {
            e.target.backgroundColor = 'transparent'
          }}
        >
          Dashboard
        </Link>
        <Link
          to="/classify"
          style={linkStyle}
          onMouseEnter={(e) => {
            e.target.backgroundColor = '#2f7a60'
            e.target.transform = 'scale(1.05)'
          }}
          onMouseLeave={(e) => {
            e.target.backgroundColor = 'transparent'
          }}
        >
          Classify
        </Link>
      </div>
    </nav>
  )
}
