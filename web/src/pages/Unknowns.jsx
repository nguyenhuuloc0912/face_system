import { useEffect, useState } from 'react'
import { fetchUnknowns } from '../api'

function fmt(iso) {
    if (!iso) return '—'
    return new Date(iso).toLocaleString('vi-VN')
}

export default function Unknowns() {
    const [rows, setRows] = useState([])
    const [total, setTotal] = useState(0)
    const [loading, setLoading] = useState(false)
    const [dateFrom, setDateFrom] = useState('')
    const [dateTo, setDateTo] = useState('')
    const [selected, setSelected] = useState(null)

    const load = async () => {
        setLoading(true)
        try {
            const res = await fetchUnknowns({ dateFrom, dateTo, limit: 500 })
            setRows(res.data ?? [])
            setTotal(res.count ?? 0)
        } catch { }
        setLoading(false)
    }

    useEffect(() => { load() }, [])

    return (
        <>
            {/* Lightbox */}
            {selected && (
                <div
                    onClick={() => setSelected(null)}
                    style={{
                        position: 'fixed', inset: 0, background: 'rgba(0,0,0,.85)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 999,
                    }}
                >
                    <img src={selected} alt="face" style={{ maxHeight: '80vh', maxWidth: '80vw', borderRadius: 12, boxShadow: '0 8px 48px rgba(0,0,0,.8)' }} />
                </div>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
                <h3 style={{ flex: 1, fontSize: 15, fontWeight: 600 }}>Unknown Faces Log</h3>
                <input className="input-field" type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} />
                <span style={{ color: 'var(--text-muted)' }}>→</span>
                <input className="input-field" type="date" value={dateTo} onChange={e => setDateTo(e.target.value)} />
                <button className="btn btn-primary" onClick={load}>Filter</button>
                <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}>{total} records</span>
            </div>

            {loading ? (
                <div className="loading">Loading...</div>
            ) : rows.length === 0 ? (
                <div className="empty-state">
                    <div className="icon">👻</div>
                    <p>Không có người lạ được phát hiện</p>
                </div>
            ) : (
                <div className="unknown-grid">
                    {rows.map((r, i) => (
                        <div className="unknown-card" key={r.id}>
                            {r.image_path ? (
                                <img
                                    src={r.image_path}
                                    alt="unknown face"
                                    onClick={() => setSelected(r.image_path)}
                                    style={{
                                        width: 80, height: 80, objectFit: 'cover',
                                        borderRadius: 10, border: '2px solid var(--accent-red)',
                                        cursor: 'pointer', display: 'block', margin: '0 auto 10px',
                                        transition: 'transform .15s',
                                    }}
                                    onMouseEnter={e => e.target.style.transform = 'scale(1.06)'}
                                    onMouseLeave={e => e.target.style.transform = 'scale(1)'}
                                />
                            ) : (
                                <div className="face-placeholder">👤</div>
                            )}
                            <div className="label">Unknown #{i + 1}</div>
                            <div className="time">{fmt(r.detected_at)}</div>
                        </div>
                    ))}
                </div>
            )}
        </>
    )
}
