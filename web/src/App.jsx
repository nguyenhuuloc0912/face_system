import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Attendance from './pages/Attendance'
import Unknowns from './pages/Unknowns'
import Settings from './pages/Settings'

const NAV = [
    { to: '/', icon: '⬡', label: 'Dashboard' },
    { to: '/attendance', icon: '✓', label: 'Attendance' },
    { to: '/unknowns', icon: '⚠', label: 'Unknowns' },
    { to: '/settings', icon: '⚙', label: 'Settings' },
]

function Sidebar() {
    return (
        <div className="sidebar">
            <div className="sidebar-logo">
                <h1>🎯 AI VIET NAM</h1>
                <span>Face Re-ID Dashboard</span>
            </div>
            <nav className="sidebar-nav">
                {NAV.map(n => (
                    <NavLink
                        key={n.to}
                        to={n.to}
                        end={n.to === '/'}
                        className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                    >
                        <span className="nav-icon">{n.icon}</span>
                        {n.label}
                    </NavLink>
                ))}
            </nav>
        </div>
    )
}

const PAGE_TITLES = {
    '/': 'Dashboard',
    '/attendance': 'Attendance Log',
    '/unknowns': 'Unknown Faces',
    '/settings': 'Settings',
}

function Header() {
    const { pathname } = useLocation()
    return (
        <div className="page-header">
            <h2>{PAGE_TITLES[pathname] ?? 'FaceID'}</h2>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="status-dot online" />
                <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>API Connected</span>
            </div>
        </div>
    )
}

export default function App() {
    return (
        <div className="app-layout">
            <Sidebar />
            <div className="main-content">
                <Header />
                <div className="page-body">
                    <Routes>
                        <Route path="/" element={<Dashboard />} />
                        <Route path="/attendance" element={<Attendance />} />
                        <Route path="/unknowns" element={<Unknowns />} />
                        <Route path="/settings" element={<Settings />} />
                    </Routes>
                </div>
            </div>
        </div>
    )
}
