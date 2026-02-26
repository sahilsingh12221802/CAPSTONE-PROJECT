import { Outlet } from 'react-router-dom'
import Navbar from '../components/Navbar'


export default function Layout() {
  const layoutStyle = {
    display: 'flex',
    flexDirection: 'column',
    minHeight: '100vh',
    backgroundColor: '#f0f2f5'
  }

  const mainStyle = {
    flex: 1,
    width: '100%'
  }

  return (
    <div style={layoutStyle}>
      <Navbar />
      <main style={mainStyle}>
        <Outlet />
      </main>
    </div>
  )
}
