import { useEffect, useState } from 'react'
import { fetchStats, fetchAttendance, fetchInferStatus, startInfer, stopInfer } from '../api'

function StatCard({ label, value, sub, color, icon }) {
    return (
        <div className={`stat-card ${color}`}>
            <div className="stat-icon">{icon}</div>
            <div className="stat-label">{label}</div>
            <div className="stat-value">{value ?? '—'}</div>
            {sub && <div className="stat-sub">{sub}</div>}
        </div>
    )
}

function FaceThumb({ src, size = 40 }) {
    const [err, setErr] = useState(false)
    if (!src || err) {
        return (
            <div style={{
                width: size, height: size, borderRadius: 6,
                background: 'var(--bg-hover)', display: 'flex',
                alignItems: 'center', justifyContent: 'center',
                fontSize: size * 0.4, color: 'var(--text-muted)',
            }}>👤</div>
        )
    }
    return (
        <img
            src={src}
            alt="face"
            onError={() => setErr(true)}
            style={{
                width: size, height: size, objectFit: 'cover',
                borderRadius: 6, border: '1px solid var(--border)',
                display: 'block',
            }}
        />
    )
}

function formatTime(iso) {
    if (!iso) return '—'
    return new Date(iso).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export default function Dashboard() {
    const [stats, setStats] = useState(null)
    const [recent, setRecent] = useState([])
    const [inferEnabled, setInferEnabled] = useState(false)
    const [loading, setLoading] = useState(true)

    const load = async () => {
        try {
            const [s, a, st] = await Promise.all([fetchStats(), fetchAttendance({ limit: 10 }), fetchInferStatus()])
            setStats(s)
            setRecent(a.data ?? [])
            setInferEnabled(st.enabled)
        } catch { }
        setLoading(false)
    }

    useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t) }, [])

    const toggleInfer = async () => {
        if (inferEnabled) await stopInfer(); else await startInfer()
        const s = await fetchInferStatus()
        setInferEnabled(s.enabled)
    }

    if (loading) return <div className="loading">Loading dashboard...</div>

    return (
        <>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
                <div>
                    <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>Overview</h3>
                    <p style={{ color: 'var(--text-secondary)', fontSize: 13 }}>Hôm nay · Tự động cập nhật mỗi 5 giây</p>
                </div>
                <button className={`btn ${inferEnabled ? 'btn-danger' : 'btn-success'}`} onClick={toggleInfer}>
                    {inferEnabled ? '⏹ Stop Inference' : '▶ Start Inference'}
                </button>
            </div>

            <div className="stat-grid">
                <StatCard label="Attendance Today" value={stats?.today_known} color="blue" icon="👤" sub={`Total: ${stats?.total_known ?? 0}`} />
                <StatCard label="Unique People" value={stats?.unique_today} color="green" icon="✓" />
                <StatCard label="Unknown Today" value={stats?.today_unknown} color="red" icon="⚠" sub={`Total: ${stats?.total_unknown ?? 0}`} />
                <StatCard label="Inference Status" value={inferEnabled ? 'ON' : 'OFF'} color={inferEnabled ? 'green' : 'red'} icon="🎯" />
            </div>

            <div className="table-card">
                <div className="table-toolbar">
                    <span className="table-title">Recent Attendance</span>
                </div>
                {recent.length === 0 ? (
                    <div className="empty-state"><div className="icon">📋</div><p>Chưa có dữ liệu hôm nay</p></div>
                ) : (
                    <table>
                        <thead><tr><th>Photo</th><th>Name</th><th>Similarity</th><th>Time</th></tr></thead>
                        <tbody>
                            {recent.map(r => (
                                <tr key={r.id}>
                                    <td><FaceThumb src={r.image_path} size={40} /></td>
                                    <td style={{ fontWeight: 600 }}>{r.name}</td>
                                    <td><span className="badge badge-green">{(r.similarity * 100).toFixed(1)}%</span></td>
                                    <td style={{ color: 'var(--text-secondary)' }}>{formatTime(r.detected_at)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </>
    )
}
