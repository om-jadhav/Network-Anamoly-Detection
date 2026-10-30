import { useState } from 'react'
import {
  Activity,
  Shield,
  LayoutDashboard,
  ChartNoAxesCombined,
  BrainCircuit,
  ScanSearch,
  Bell,
  Search,
  ChevronDown,
  ArrowUpRight,
  ArrowDownRight,
  Network,
  ShieldAlert,
  CheckCircle2,
  Clock3,
  Menu,
  X,
  Zap,
  LockKeyhole,
} from 'lucide-react'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from 'recharts'

const navigation = [
  { name: 'Overview', icon: LayoutDashboard },
  { name: 'Attack Analytics', icon: ChartNoAxesCombined },
  { name: 'Model Performance', icon: BrainCircuit },
  { name: 'Traffic Prediction', icon: ScanSearch },
]

const trafficData = [
  { day: 'Mon', normal: 65, attacks: 22 },
  { day: 'Tue', normal: 78, attacks: 30 },
  { day: 'Wed', normal: 55, attacks: 18 },
  { day: 'Thu', normal: 88, attacks: 36 },
  { day: 'Fri', normal: 72, attacks: 25 },
  { day: 'Sat', normal: 95, attacks: 42 },
  { day: 'Sun', normal: 82, attacks: 29 },
]

const distributionData = [
  { name: 'Normal', value: 80.33, color: '#22d3a6' },
  { name: 'Attack', value: 19.67, color: '#f87171' },
]

const attackData = [
  { name: 'DDoS', recall: 99.96 },
  { name: 'PortScan', recall: 99.99 },
  { name: 'Bot', recall: 84.53 },
  { name: 'Infiltration', recall: 28.57 },
  { name: 'Web XSS', recall: 99.15 },
]

const detections = [
  {
    id: 'EVT-001',
    type: 'DDoS',
    source: '192.168.1.105',
    destination: '10.0.0.24',
    status: 'Attack',
    confidence: '99.8%',
    time: 'Historical sample',
  },
  {
    id: 'EVT-002',
    type: 'Normal Traffic',
    source: '192.168.1.110',
    destination: '10.0.0.12',
    status: 'Normal',
    confidence: '98.2%',
    time: 'Historical sample',
  },
  {
    id: 'EVT-003',
    type: 'PortScan',
    source: '172.16.0.18',
    destination: '10.0.0.21',
    status: 'Attack',
    confidence: '99.1%',
    time: 'Historical sample',
  },
  {
    id: 'EVT-004',
    type: 'Bot',
    source: '192.168.1.125',
    destination: '10.0.0.31',
    status: 'Attack',
    confidence: '76.4%',
    time: 'Historical sample',
  },
]

function MetricCard({ title, value, subtitle, icon: Icon, accent, trend }) {
  return (
    <div className="rounded-2xl border border-white/[0.07] bg-[#101720] p-5 transition hover:border-white/[0.15]">
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-400">{title}</span>
        <div className={`rounded-xl p-2.5 ${accent}`}>
          <Icon size={19} />
        </div>
      </div>
      <div className="mt-5 flex items-end justify-between gap-3">
        <div>
          <h3 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
            {value}
          </h3>
          <p className="mt-2 text-xs text-slate-500">{subtitle}</p>
        </div>
        {trend && (
          <span className="mb-1 flex items-center gap-1 text-xs text-emerald-400">
            <ArrowUpRight size={14} />
            {trend}
          </span>
        )}
      </div>
    </div>
  )
}

function SectionHeading({ title, description, action }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        <p className="mt-1 text-sm text-slate-500">{description}</p>
      </div>
      {action && (
        <span className="rounded-lg border border-white/[0.08] px-3 py-2 text-xs text-slate-400">
          {action}
        </span>
      )}
    </div>
  )
}

function OverviewPage() {
  return (
    <div className="space-y-7">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.2em] text-emerald-400">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Dataset evaluation
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
            Security Overview
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
            Network anomaly detection insights from the CIC-IDS2017 evaluation.
          </p>
        </div>
        <div className="flex items-center gap-2 self-start rounded-xl border border-emerald-400/15 bg-emerald-400/[0.06] px-3 py-2 text-xs text-emerald-300 sm:self-auto">
          <CheckCircle2 size={15} />
          Model artifacts available
        </div>
      </div>

      <div className="rounded-xl border border-amber-400/15 bg-amber-400/[0.05] px-4 py-3 text-xs leading-5 text-amber-200/80">
        <strong className="font-medium text-amber-200">Evaluation mode:</strong>{' '}
        Statistics and visualizations represent historical dataset results and illustrative UI samples. Live traffic monitoring and API predictions are not connected yet.
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Evaluated Network Flows"
          value="562,428"
          subtitle="Held-out evaluation records"
          icon={Network}
          accent="bg-sky-400/10 text-sky-300"
        />
        <MetricCard
          title="Test Accuracy"
          value="99.79%"
          subtitle="Historical test evaluation"
          icon={Shield}
          accent="bg-emerald-400/10 text-emerald-300"
        />
        <MetricCard
          title="Attack Recall"
          value="99.92%"
          subtitle="At the evaluated 0.20 threshold"
          icon={ShieldAlert}
          accent="bg-violet-400/10 text-violet-300"
        />
        <MetricCard
          title="Benign False Positive Rate"
          value="0.24%"
          subtitle="1,072 false positives"
          icon={Activity}
          accent="bg-rose-400/10 text-rose-300"
        />
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.65fr_1fr]">
        <div className="rounded-2xl border border-white/[0.07] bg-[#101720] p-5 sm:p-6">
          <SectionHeading
            title="Traffic Activity"
            description="Illustrative weekly traffic pattern"
            action="Sample visualization"
          />
          <div className="h-[270px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trafficData} margin={{ top: 10, right: 5, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="normalGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22d3a6" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="#22d3a6" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="attackGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f87171" stopOpacity={0.2} />
                    <stop offset="100%" stopColor="#f87171" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#ffffff0c" vertical={false} />
                <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    background: '#151e29',
                    border: '1px solid #ffffff18',
                    borderRadius: 12,
                    color: '#e2e8f0',
                  }}
                />
                <Area type="monotone" dataKey="normal" name="Normal (illustrative)" stroke="#22d3a6" fill="url(#normalGradient)" strokeWidth={2} />
                <Area type="monotone" dataKey="attacks" name="Attacks (illustrative)" stroke="#f87171" fill="url(#attackGradient)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 flex gap-5 text-xs text-slate-400">
            <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-emerald-400" />Normal traffic</span>
            <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-rose-400" />Attack traffic</span>
          </div>
        </div>

        <div className="rounded-2xl border border-white/[0.07] bg-[#101720] p-5 sm:p-6">
          <SectionHeading
            title="Evaluation Class Balance"
            description="Distribution in the test dataset"
          />
          <div className="relative h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={distributionData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={65} outerRadius={90} paddingAngle={4} stroke="none">
                  {distributionData.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value) => [`${Number(value).toFixed(2)}%`, 'Share']}
                  contentStyle={{ background: '#151e29', border: '1px solid #ffffff18', borderRadius: 12 }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-2xl font-semibold text-white">562K</span>
              <span className="mt-1 text-xs text-slate-500">Total records</span>
            </div>
          </div>
          <div className="space-y-4">
            {distributionData.map((item) => (
              <div key={item.name} className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2 text-slate-400">
                  <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: item.color }} />
                  {item.name}
                </span>
                <span className="font-medium text-white">{item.value}%</span>
              </div>
            ))}
          </div>
          <p className="mt-4 border-t border-white/[0.06] pt-4 text-xs leading-5 text-slate-500">
            Class balance is calculated from the historical test-set confusion matrix support.
          </p>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.25fr_1fr]">
        <div className="rounded-2xl border border-white/[0.07] bg-[#101720] p-5 sm:p-6">
          <SectionHeading
            title="Recall by Attack Category"
            description="Selected historical test-set results"
            action="Model evaluation"
          />
          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={attackData} layout="vertical" margin={{ top: 0, right: 20, left: 10, bottom: 0 }}>
                <CartesianGrid stroke="#ffffff0c" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" width={85} tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
                <Tooltip
                  formatter={(value) => [`${value}%`, 'Recall']}
                  contentStyle={{ background: '#151e29', border: '1px solid #ffffff18', borderRadius: 12 }}
                />
                <Bar dataKey="recall" name="Recall" fill="#34d399" radius={[0, 5, 5, 0]} barSize={17} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-xs leading-5 text-slate-500">
            Small-support categories can have unstable recall estimates; results are not a guarantee of real-world performance.
          </p>
        </div>

        <div className="rounded-2xl border border-white/[0.07] bg-[#101720] p-5 sm:p-6">
          <SectionHeading
            title="Model Evaluation"
            description="Binary classification at threshold 0.20"
          />
          <div className="space-y-5">
            {[
              { label: 'Accuracy', value: 99.79, color: '#38bdf8' },
              { label: 'Attack Precision', value: 99.01, color: '#a78bfa' },
              { label: 'Attack Recall', value: 99.92, color: '#34d399' },
              { label: 'Attack F1-score', value: 99.46, color: '#fbbf24' },
            ].map((metric) => (
              <div key={metric.label}>
                <div className="mb-2 flex items-center justify-between text-sm">
                  <span className="text-slate-400">{metric.label}</span>
                  <span className="font-medium text-white">{metric.value}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-white/[0.06]">
                  <div className="h-full rounded-full" style={{ width: `${metric.value}%`, backgroundColor: metric.color }} />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-6 rounded-xl border border-white/[0.06] bg-[#0b1118] p-4">
            <div className="flex items-center gap-2 text-sm font-medium text-white">
              <LockKeyhole size={16} className="text-emerald-400" />
              Evaluation context
            </div>
            <p className="mt-2 text-xs leading-5 text-slate-500">
              These values summarize the saved tree-model evaluation. The UI is not currently sending data to the model.
            </p>
          </div>
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-white/[0.07] bg-[#101720]">
        <div className="p-5 sm:p-6">
          <SectionHeading
            title="Detection Examples"
            description="Illustrative rows for the dashboard interface—not actual prediction logs"
            action="Demo data"
          />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[700px] text-left text-sm">
            <thead className="border-y border-white/[0.06] bg-white/[0.02] text-xs uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-6 py-4 font-medium">Event ID</th>
                <th className="px-6 py-4 font-medium">Traffic Type</th>
                <th className="px-6 py-4 font-medium">Source</th>
                <th className="px-6 py-4 font-medium">Destination</th>
                <th className="px-6 py-4 font-medium">Status</th>
                <th className="px-6 py-4 font-medium">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05]">
              {detections.map((item) => (
                <tr key={item.id} className="transition hover:bg-white/[0.02]">
                  <td className="px-6 py-4 font-mono text-xs text-slate-400">{item.id}</td>
                  <td className="px-6 py-4 text-slate-200">{item.type}</td>
                  <td className="px-6 py-4 font-mono text-xs text-slate-400">{item.source}</td>
                  <td className="px-6 py-4 font-mono text-xs text-slate-400">{item.destination}</td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ${item.status === 'Attack'
                        ? 'bg-rose-400/10 text-rose-300'
                        : 'bg-emerald-400/10 text-emerald-300'
                      }`}>
                      <span className={`h-1.5 w-1.5 rounded-full ${item.status === 'Attack' ? 'bg-rose-400' : 'bg-emerald-400'}`} />
                      {item.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 font-medium text-slate-300">{item.confidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="border-t border-white/[0.06] px-6 py-3 text-xs text-slate-500">
          Demo interface only · {detections.length} illustrative entries
        </div>
      </div>
    </div>
  )
}

function PlaceholderPage({ page }) {
  const details = {
    'Attack Analytics': {
      icon: ChartNoAxesCombined,
      title: 'Attack Analytics',
      description: 'Explore attack categories and their historical detection performance.',
    },
    'Model Performance': {
      icon: BrainCircuit,
      title: 'Model Performance',
      description: 'Review the trained model’s evaluation metrics and classification results.',
    },
    'Traffic Prediction': {
      icon: ScanSearch,
      title: 'Traffic Prediction',
      description: 'Prepare a network-flow feature submission interface for the FastAPI model endpoint.',
    },
  }

  const item = details[page] ?? details['Attack Analytics']
  const Icon = item.icon

  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <div className="max-w-md text-center">
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl border border-emerald-400/15 bg-emerald-400/[0.06] text-emerald-300">
          <Icon size={28} />
        </div>
        <p className="mt-6 text-xs font-medium uppercase tracking-[0.2em] text-emerald-400">
          Dashboard module
        </p>
        <h1 className="mt-3 text-2xl font-semibold text-white">{item.title}</h1>
        <p className="mt-3 text-sm leading-6 text-slate-400">{item.description}</p>
        <div className="mt-6 inline-flex items-center gap-2 rounded-lg border border-white/[0.08] bg-[#101720] px-4 py-3 text-xs text-slate-400">
          <Clock3 size={15} />
          Detailed module coming next
        </div>
      </div>
    </div>
  )
}
function ModelPerformancePage() {
  const metrics = [
    {
      label: 'Test Accuracy',
      value: '99.79%',
      description: 'Overall prediction accuracy',
      color: 'text-emerald-400',
    },
    {
      label: 'Attack Precision',
      value: '99.01%',
      description: 'Predicted attacks that were attacks',
      color: 'text-sky-400',
    },
    {
      label: 'Attack Recall',
      value: '99.92%',
      description: 'Attacks correctly detected',
      color: 'text-violet-400',
    },
    {
      label: 'Attack F1-score',
      value: '99.46%',
      description: 'Balance of precision and recall',
      color: 'text-amber-400',
    },
  ]

  const confusionMatrix = [
    { actual: 'Normal', normal: 453760, attack: 1072 },
    { actual: 'Attack', normal: 85, attack: 107511 },
  ]

  const attackPerformance = [
    { name: 'DDoS', recall: 99.96 },
    { name: 'PortScan', recall: 99.99 },
    { name: 'Bot', recall: 84.53 },
    { name: 'Infiltration', recall: 28.57 },
    { name: 'Web XSS', recall: 99.15 },
  ]

  return (
    <div className="space-y-7">
      {/* Page heading */}
      <div>
        <div className="mb-3 flex items-center gap-2 text-xs text-emerald-400">
          <BrainCircuit size={15} />
          <span>MODEL EVALUATION</span>
        </div>

        <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
          Model Performance
        </h1>

        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
          Evaluation results for the network anomaly detection model on the
          CIC-IDS2017 test dataset.
        </p>
      </div>

      {/* Evaluation notice */}
      <div className="flex flex-col gap-2 rounded-xl border border-amber-400/20 bg-amber-400/[0.05] p-4 sm:flex-row sm:items-center">
        <Clock3 size={19} className="shrink-0 text-amber-400" />
        <p className="text-xs leading-5 text-slate-300">
          <span className="font-semibold text-amber-300">
            Offline evaluation
          </span>
          {' '}· These are historical test results at a 0.20 prediction
          threshold, not live network traffic predictions.
        </p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {metrics.map((metric) => (
          <div
            key={metric.label}
            className="rounded-2xl border border-white/[0.07] bg-[#101720] p-5"
          >
            <p className="text-sm text-slate-400">{metric.label}</p>

            <p className={`mt-4 text-3xl font-semibold tracking-tight ${metric.color}`}>
              {metric.value}
            </p>

            <p className="mt-3 text-xs leading-5 text-slate-500">
              {metric.description}
            </p>
          </div>
        ))}
      </div>

      {/* Confusion matrix */}
      <section className="rounded-2xl border border-white/[0.07] bg-[#101720] p-5 sm:p-6">
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-white">
            Confusion Matrix
          </h2>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            Actual labels compared with the model's predicted labels.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[440px] border-separate border-spacing-2 text-center text-sm">
            <thead>
              <tr>
                <th className="p-3 text-left text-xs font-medium text-slate-500">
                  Actual / Predicted
                </th>
                <th className="rounded-lg bg-white/[0.03] p-3 text-slate-300">
                  Normal
                </th>
                <th className="rounded-lg bg-white/[0.03] p-3 text-slate-300">
                  Attack
                </th>
              </tr>
            </thead>

            <tbody>
              {confusionMatrix.map((row) => (
                <tr key={row.actual}>
                  <th className="p-3 text-left font-medium text-slate-300">
                    {row.actual}
                  </th>

                  <td className="rounded-xl border border-emerald-400/10 bg-emerald-400/[0.08] p-5">
                    <p className="text-xl font-semibold text-emerald-300">
                      {row.normal.toLocaleString('en-US')}
                    </p>
                    <p className="mt-1 text-[10px] text-slate-500">
                      {row.actual === 'Normal'
                        ? 'True Negative'
                        : 'False Negative'}
                    </p>
                  </td>

                  <td className="rounded-xl border border-rose-400/10 bg-rose-400/[0.07] p-5">
                    <p className="text-xl font-semibold text-rose-300">
                      {row.attack.toLocaleString('en-US')}
                    </p>
                    <p className="mt-1 text-[10px] text-slate-500">
                      {row.actual === 'Normal'
                        ? 'False Positive'
                        : 'True Positive'}
                    </p>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="mt-4 text-xs leading-5 text-slate-500">
          False positives are normal flows flagged as attacks. False negatives
          are attacks incorrectly classified as normal.
        </p>
      </section>

      {/* Attack-wise recall */}
      <section className="rounded-2xl border border-white/[0.07] bg-[#101720] p-5 sm:p-6">
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-white">
            Attack-wise Detection Recall
          </h2>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            Recall varies by attack category. Higher recall means fewer attacks
            of that category were missed.
          </p>
        </div>

        <div className="space-y-6">
          {attackPerformance.map((attack) => (
            <div key={attack.name}>
              <div className="mb-2 flex items-center justify-between gap-4">
                <span className="text-sm text-slate-300">
                  {attack.name}
                </span>
                <span className={`text-sm font-semibold ${
                  attack.recall < 60
                    ? 'text-rose-400'
                    : attack.recall < 90
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                }`}>
                  {attack.recall.toFixed(2)}%
                </span>
              </div>

              <div
                className="h-2.5 overflow-hidden rounded-full bg-white/[0.06]"
                role="progressbar"
                aria-label={`${attack.name} recall`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={attack.recall}
              >
                <div
                  className={`h-full rounded-full transition-all ${
                    attack.recall < 60
                      ? 'bg-rose-400'
                      : attack.recall < 90
                        ? 'bg-amber-400'
                        : 'bg-emerald-400'
                  }`}
                  style={{ width: `${attack.recall}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 rounded-xl border border-rose-400/15 bg-rose-400/[0.04] p-4">
          <div className="flex items-start gap-3">
            <ShieldAlert size={18} className="mt-0.5 shrink-0 text-rose-400" />
            <div>
              <p className="text-sm font-medium text-rose-300">
                Important finding: Infiltration and Bot detection
              </p>
              <p className="mt-2 text-xs leading-5 text-slate-400">
                In this evaluation, Infiltration recall was 28.57% and Bot
                recall was 84.53%. These categories deserve further
                investigation and model improvement.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Additional metrics */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="rounded-2xl border border-white/[0.07] bg-[#101720] p-5">
          <p className="text-sm text-slate-400">Evaluated test flows</p>
          <p className="mt-3 text-2xl font-semibold text-white">562,428</p>
          <p className="mt-2 text-xs text-slate-500">
            Total records in the held-out test evaluation.
          </p>
        </div>

        <div className="rounded-2xl border border-white/[0.07] bg-[#101720] p-5">
          <p className="text-sm text-slate-400">Benign false-positive rate</p>
          <p className="mt-3 text-2xl font-semibold text-amber-300">0.24%</p>
          <p className="mt-2 text-xs text-slate-500">
            1,072 normal flows were incorrectly flagged as attacks.
          </p>
        </div>
      </section>
    </div>
  )
}

export default function App() {
  const [activePage, setActivePage] = useState('Overview')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')

const pageContent = activePage === 'Overview'
  ? <OverviewPage />
  : activePage === 'Model Performance'
    ? <ModelPerformancePage />
    : <PlaceholderPage page={activePage} />

  return (
    <div className="min-h-screen bg-[#0b1118] text-slate-200">
      {sidebarOpen && (
        <button
          aria-label="Close navigation overlay"
          className="fixed inset-0 z-40 bg-black/60 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`fixed inset-y-0 left-0 z-50 flex w-[260px] flex-col border-r border-white/[0.07] bg-[#0e151e] transition-transform duration-200 lg:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}>
        <div className="flex h-[76px] items-center gap-3 border-b border-white/[0.07] px-6">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-400 text-[#07120f]">
            <Shield size={22} strokeWidth={2.5} />
          </div>
          <div>
            <h2 className="text-sm font-semibold tracking-wide text-white">NEXUS<span className="text-emerald-400">SEC</span></h2>
            <p className="mt-0.5 text-[10px] uppercase tracking-[0.18em] text-slate-500">Network intelligence</p>
          </div>
          <button
            className="ml-auto rounded-lg p-2 text-slate-400 hover:bg-white/5 lg:hidden"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X size={18} />
          </button>
        </div>

        <div className="px-4 pt-7">
          <p className="px-3 pb-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-600">
            Workspace
          </p>
          <nav className="space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon
              const active = activePage === item.name

              return (
                <button
                  key={item.name}
                  onClick={() => {
                    setActivePage(item.name)
                    setSidebarOpen(false)
                  }}
                  className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm transition ${active
                      ? 'bg-emerald-400/[0.10] font-medium text-emerald-300'
                      : 'text-slate-400 hover:bg-white/[0.04] hover:text-slate-200'
                    }`}
                >
                  <Icon size={18} />
                  <span>{item.name}</span>
                  {active && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-emerald-400" />}
                </button>
              )
            })}
          </nav>
        </div>

        <div className="mt-auto p-4">
          <div className="rounded-2xl border border-white/[0.07] bg-[#111b25] p-4">
            <div className="flex items-center gap-2 text-xs font-medium text-slate-200">
              <Zap size={15} className="text-emerald-400" />
              Model workspace
            </div>
            <p className="mt-2 text-xs leading-5 text-slate-500">
              CIC-IDS2017 · Binary classification
            </p>
            <div className="mt-4 flex items-center justify-between border-t border-white/[0.06] pt-3">
              <span className="text-[11px] text-slate-500">API connection</span>
              <span className="flex items-center gap-1.5 text-[10px] text-amber-300">
                <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                Not connected
              </span>
            </div>
          </div>
          <div className="mt-4 flex items-center gap-3 px-2 py-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-slate-800 text-xs font-semibold text-white">
              NJ
            </div>
            <div className="min-w-0">
              <p className="text-xs font-medium text-slate-200">Network Analyst</p>
              <p className="mt-1 text-[10px] text-slate-500">Project workspace</p>
            </div>
          </div>
        </div>
      </aside>

      <div className="min-h-screen lg:pl-[260px]">
        <header className="sticky top-0 z-30 flex h-[76px] items-center justify-between gap-4 border-b border-white/[0.07] bg-[#0b1118]/90 px-4 backdrop-blur-xl sm:px-7 lg:px-9">
          <div className="flex min-w-0 items-center gap-3">
            <button
              onClick={() => setSidebarOpen(true)}
              className="rounded-lg border border-white/10 p-2 text-slate-300 lg:hidden"
              aria-label="Open navigation"
            >
              <Menu size={19} />
            </button>
            <div className="hidden items-center gap-2 text-xs text-slate-500 sm:flex">
              <LayoutDashboard size={15} />
              <span>Workspace</span>
              <span>/</span>
            </div>
            <span className="truncate text-sm font-medium text-white">{activePage}</span>
          </div>

          <div className="flex items-center gap-3">
            <label className="hidden items-center gap-2 rounded-xl border border-white/[0.08] bg-white/[0.02] px-3 py-2 md:flex">
              <Search size={15} className="text-slate-500" />
              <input
                value={searchTerm}
                onChange={(event) => setSearchTerm(event.target.value)}
                placeholder="Search dashboard..."
                className="w-36 bg-transparent text-xs text-white outline-none placeholder:text-slate-600"
              />
              <kbd className="rounded border border-white/10 px-1.5 py-0.5 text-[10px] text-slate-500">/</kbd>
            </label>
            <button
              className="relative rounded-xl border border-white/[0.08] p-2.5 text-slate-400 transition hover:bg-white/[0.04] hover:text-white"
              aria-label="Notifications"
              title="Notifications are not connected"
            >
              <Bell size={18} />
              <span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-emerald-400" />
            </button>
            <button className="hidden items-center gap-2 rounded-xl border border-white/[0.08] px-3 py-2 text-xs text-slate-300 sm:flex">
              CIC-IDS2017
              <ChevronDown size={14} />
            </button>
          </div>
        </header>

        <main className="mx-auto max-w-[1600px] p-4 sm:p-7 lg:p-9">
          {searchTerm.trim() ? (
            <div className="mb-5 rounded-xl border border-white/[0.08] bg-[#101720] p-4 text-sm text-slate-400">
              Search is a UI placeholder. Backend search will be added later.
            </div>
          ) : null}
          {pageContent}
          <footer className="mt-10 flex flex-col justify-between gap-2 border-t border-white/[0.07] py-5 text-xs text-slate-600 sm:flex-row">
            <span>NEXUSSEC · Network Anomaly Detection</span>
            <span>Research prototype · CIC-IDS2017 evaluation</span>
          </footer>
        </main>
      </div>
    </div>
  )
}