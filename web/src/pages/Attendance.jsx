import { useEffect, useState } from 'react'
import { fetchAttendance } from '../api'

function FaceThumb({ src, size = 48 }) {
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
            style={{ width: size, height: size, objectFit: 'cover', borderRadius: 6, border: '1px solid var(--border)' }}
        />
    )
}

function fmt(iso) {
    if (!iso) return '—'
    return new Date(iso).toLocaleString('vi-VN')
}

export default function Attendance() {
    const [rows, setRows] = useState([])
    const [total, setTotal] = useState(0)
    const [loading, setLoading] = useState(false)
    const [name, setName] = useState('')
    const [dateFrom, setDateFrom] = useState('')
    const [dateTo, setDateTo] = useState('')

    const load = async () => {
        setLoading(true)
        try {
            const res = await fetchAttendance({ name, dateFrom, dateTo, limit: 500 })
            setRows(res.data ?? [])
            setTotal(res.count ?? 0)
        } catch { }
        setLoading(false)
    }

    useEffect(() => { load() }, [])

    const handleSearch = e => { e.preventDefault(); load() }
    const handleReset = () => { setName(''); setDateFrom(''); setDateTo(''); setTimeout(load, 0) }

    return (
        <div className="table-card">
            <div className="table-toolbar">
                <span className="table-title">Attendance Log</span>
                <form onSubmit={handleSearch} style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                    <input className="input-field" placeholder="Search by name..." value={name}
                        onChange={e => setName(e.target.value)} style={{ width: 180 }} />
                    <input className="input-field" type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} />
                    <span style={{ color: 'var(--text-muted)' }}>→</span>
                    <input className="input-field" type="date" value={dateTo} onChange={e => setDateTo(e.target.value)} />
                    <button className="btn btn-primary" type="submit">Search</button>
                    <button className="btn btn-ghost" type="button" onClick={handleReset}>Reset</button>
                </form>
                <span style={{ marginLeft: 'auto', color: 'var(--text-secondary)', fontSize: 12 }}>{total} records</span>
            </div>

            {loading ? (
                <div className="loading">Loading...</div>
            ) : rows.length === 0 ? (
                <div className="empty-state"><div className="icon">🔍</div><p>Không tìm thấy dữ liệu</p></div>
            ) : (
                <div style={{ overflowX: 'auto', maxHeight: 'calc(100vh - 260px)', overflowY: 'auto' }}>
                    <table>
                        <thead>
                            <tr>
                                <th style={{ width: 60 }}>Photo</th>
                                <th>#</th>
                                <th>Name</th>
                                <th>Similarity</th>
                                <th>Detected At</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((r, i) => (
                                <tr key={r.id}>
                                    <td><FaceThumb src={r.image_path} size={48} /></td>
                                    <td style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                                    <td style={{ fontWeight: 600 }}>{r.name}</td>
                                    <td><span className="badge badge-green">{(r.similarity * 100).toFixed(1)}%</span></td>
                                    <td style={{ color: 'var(--text-secondary)' }}>{fmt(r.detected_at)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}
