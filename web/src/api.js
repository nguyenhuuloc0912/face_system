const BASE = '/api'

export async function fetchStats() {
    const res = await fetch(`${BASE}/stats`)
    if (!res.ok) throw new Error('Failed to fetch stats')
    return res.json()
}

export async function fetchAttendance({ name = '', dateFrom = '', dateTo = '', limit = 200 } = {}) {
    const p = new URLSearchParams()
    if (name) p.set('name', name)
    if (dateFrom) p.set('date_from', dateFrom)
    if (dateTo) p.set('date_to', dateTo)
    p.set('limit', limit)
    const res = await fetch(`${BASE}/attendance?${p}`)
    if (!res.ok) throw new Error('Failed to fetch attendance')
    return res.json()
}

export async function fetchUnknowns({ dateFrom = '', dateTo = '', limit = 200 } = {}) {
    const p = new URLSearchParams()
    if (dateFrom) p.set('date_from', dateFrom)
    if (dateTo) p.set('date_to', dateTo)
    p.set('limit', limit)
    const res = await fetch(`${BASE}/unknowns?${p}`)
    if (!res.ok) throw new Error('Failed to fetch unknowns')
    return res.json()
}

export async function fetchSettings() {
    const res = await fetch(`${BASE}/settings`)
    if (!res.ok) throw new Error('Failed to fetch settings')
    return res.json()
}

export async function fetchModels() {
    const res = await fetch(`${BASE}/models`)
    if (!res.ok) throw new Error('Failed to fetch models')
    return res.json()
}

export async function saveSettings(payload) {
    const res = await fetch(`${BASE}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    })
    if (!res.ok) throw new Error('Failed to save settings')
    return res.json()
}

export async function fetchInferStatus() {
    const res = await fetch(`${BASE}/infer/status`)
    if (!res.ok) throw new Error('Failed to fetch infer status')
    return res.json()
}

export async function startInfer() {
    const res = await fetch(`${BASE}/infer/start`, { method: 'POST' })
    return res.json()
}

export async function stopInfer() {
    const res = await fetch(`${BASE}/infer/stop`, { method: 'POST' })
    return res.json()
}
