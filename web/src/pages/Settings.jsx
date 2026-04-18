import { useEffect, useState } from 'react'
import { fetchModels, fetchSettings, saveSettings } from '../api'

export default function Settings() {
    const [form, setForm] = useState({
        det_weight: '',
        rec_weight: '',
        confidence_thresh: '',
        similarity_thresh: '',
        unknown_debounce_sec: '',
        known_debounce_min: '',
    })
    const [models, setModels] = useState([])
    const [loading, setLoading] = useState(true)
    const [saved, setSaved] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        Promise.all([fetchSettings(), fetchModels()])
            .then(([d, modelResponse]) => {
                setForm({
                    det_weight: d.det_weight ?? '',
                    rec_weight: d.rec_weight ?? '',
                    confidence_thresh: d.confidence_thresh ?? '',
                    similarity_thresh: d.similarity_thresh ?? '',
                    unknown_debounce_sec: d.unknown_debounce_sec ?? '',
                    known_debounce_min: d.known_debounce_min ?? '',
                })
                setModels(modelResponse.models ?? [])
                setLoading(false)
            })
            .catch(() => { setError('Cannot connect to API'); setLoading(false) })
    }, [])

    const handleChange = e => setForm(f => ({ ...f, [e.target.name]: e.target.value }))

    const handleSubmit = async e => {
        e.preventDefault()
        setError('')
        try {
            await saveSettings({
                det_weight: form.det_weight || null,
                rec_weight: form.rec_weight || null,
                confidence_thresh: form.confidence_thresh ? parseFloat(form.confidence_thresh) : null,
                similarity_thresh: form.similarity_thresh ? parseFloat(form.similarity_thresh) : null,
                unknown_debounce_sec: form.unknown_debounce_sec ? parseInt(form.unknown_debounce_sec) : null,
                known_debounce_min: form.known_debounce_min ? parseInt(form.known_debounce_min) : null,
            })
            setSaved(true)
            setTimeout(() => setSaved(false), 2500)
        } catch (err) {
            setError(err.message)
        }
    }

    const detectionModels = models
        .filter(name => name.includes('det') || name.includes('yolo') || name.includes('scrfd'))
        .map(name => `./weights/${name}`)

    const recognitionModels = models
        .filter(name => name.includes('w600k') || name.includes('arcface') || name.includes('glint'))
        .map(name => `./weights/${name}`)

    const FIELDS = [
        { name: 'det_weight', label: 'Detection Model Weight', type: 'text', ph: './weights/det_10g.onnx', list: 'det-models' },
        { name: 'rec_weight', label: 'Recognition Model Weight', type: 'text', ph: './weights/w600k_mbf.onnx', list: 'rec-models' },
        { name: 'confidence_thresh', label: 'Confidence Threshold', type: 'number', ph: '0.5', step: '0.01', min: '0', max: '1' },
        { name: 'similarity_thresh', label: 'Similarity Threshold', type: 'number', ph: '0.4', step: '0.01', min: '0', max: '1' },
        { name: 'unknown_debounce_sec', label: 'Unknown Debounce (sec)', type: 'number', ph: '5', min: '1' },
        { name: 'known_debounce_min', label: 'Known Debounce (min)', type: 'number', ph: '1', min: '1' },
    ]

    if (loading) return <div className="loading">Loading settings...</div>

    return (
        <div className="settings-card">
            <h3>⚙ Application Settings</h3>
            {error && (
                <div style={{ background: 'rgba(248,81,73,.12)', border: '1px solid var(--accent-red)', borderRadius: 6, padding: '10px 14px', marginBottom: 16, color: 'var(--accent-red)', fontSize: 13 }}>
                    {error}
                </div>
            )}
            {saved && (
                <div style={{ background: 'rgba(63,185,80,.12)', border: '1px solid var(--accent-green)', borderRadius: 6, padding: '10px 14px', marginBottom: 16, color: 'var(--accent-green)', fontSize: 13 }}>
                    ✓ Settings saved successfully!
                </div>
            )}
            <form onSubmit={handleSubmit}>
                <datalist id="det-models">
                    {detectionModels.map(model => <option key={model} value={model} />)}
                </datalist>
                <datalist id="rec-models">
                    {recognitionModels.map(model => <option key={model} value={model} />)}
                </datalist>
                {FIELDS.map(f => (
                    <div className="form-group" key={f.name}>
                        <label htmlFor={f.name}>{f.label}</label>
                        <input
                            id={f.name}
                            name={f.name}
                            className="input-field"
                            type={f.type}
                            step={f.step}
                            min={f.min}
                            max={f.max}
                            placeholder={f.ph}
                            list={f.list}
                            value={form[f.name]}
                            onChange={handleChange}
                            style={{ width: '100%' }}
                        />
                    </div>
                ))}
                <div className="form-actions">
                    <button className="btn btn-primary" type="submit">Save Settings</button>
                </div>
            </form>
        </div>
    )
}
