<template>
  <div class="llm-spend" :class="{ exhausted: spend.exhausted }" :title="tooltip">
    <span class="spend-title">LLM SPEND</span>
    <div v-for="meter in meters" :key="meter.label" class="spend-meter" :class="meter.level">
      <span class="meter-label">{{ meter.label }}</span>
      <span class="meter-value mono">{{ money(meter.used) }}<span class="meter-limit">{{ meter.limit ? ` / ${money(meter.limit)}` : ' / no limit' }}</span></span>
      <span v-if="meter.limit" class="meter-bar"><span class="meter-fill" :style="{ width: `${meter.percent}%` }"></span></span>
    </div>
    <span v-if="spend.exhausted" class="limit-chip">LIMIT REACHED</span>
    <span v-else class="spend-calls mono">{{ spend.calls }} {{ spend.calls === 1 ? 'call' : 'calls' }}</span>
  </div>
</template>

<script setup>
import { computed } from 'vue'

// Shows the llm_spend block of run-status: this simulation's agent LLM spend and
// today's app-wide LLM spend, each against its limit (a null limit means disabled).
const props = defineProps({
  spend: { type: Object, required: true }
})

const money = (usd) => (usd > 0 && usd < 0.01 ? '<$0.01' : `$${(usd || 0).toFixed(2)}`)

const meter = (label, used, limit) => {
  const percent = limit ? Math.min(100, (used / limit) * 100) : 0
  const level = !limit ? 'ok' : used >= limit ? 'over' : percent >= 80 ? 'warn' : 'ok'
  return { label, used, limit, percent, level }
}

const meters = computed(() => [
  meter('SIM', props.spend.usd, props.spend.limit_usd),
  meter('TODAY', props.spend.day_usd, props.spend.daily_limit_usd)
])

const compact = new Intl.NumberFormat('en', { notation: 'compact' })

const tooltip = computed(() => {
  const s = props.spend
  const of = (limit) => (limit ? ` of ${money(limit)}` : ' (no limit)')
  const lines = [
    `This simulation's agents: ${money(s.usd)}${of(s.limit_usd)}`,
    `All LLM calls today (UTC): ${money(s.day_usd)}${of(s.daily_limit_usd)}`,
    `${s.calls} calls · ${compact.format(s.input_tokens || 0)} input / ${compact.format(s.output_tokens || 0)} output tokens`
  ]
  if (s.reason) lines.push(s.reason)
  return lines.join('\n')
})
</script>

<style scoped>
.llm-spend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 14px;
  padding: 6px 24px;
  background: #FAFAFA;
  border-bottom: 1px solid #EAEAEA;
  white-space: nowrap;
  flex-shrink: 0;
}

.llm-spend.exhausted {
  background: #FFF5F5;
  border-bottom-color: #F8D0CC;
}

.spend-title {
  font-size: 9px;
  font-weight: 700;
  color: #000;
  letter-spacing: 0.08em;
}

.spend-meter {
  display: flex;
  align-items: center;
  gap: 6px;
}

.meter-label {
  font-size: 8px;
  font-weight: 600;
  color: #999;
  letter-spacing: 0.05em;
}

.meter-value {
  font-size: 11px;
  font-weight: 600;
  color: #333;
}

.meter-limit {
  font-size: 9px;
  font-weight: 400;
  color: #999;
}

.meter-bar {
  width: 40px;
  height: 3px;
  background: #EAEAEA;
  border-radius: 2px;
  overflow: hidden;
}

.meter-fill {
  display: block;
  height: 100%;
  background: #000;
  transition: width 0.3s ease;
}

.spend-meter.warn .meter-value { color: #D84315; }
.spend-meter.warn .meter-fill { background: #FF5722; }
.spend-meter.over .meter-value { color: #C62828; }
.spend-meter.over .meter-fill { background: #F44336; }

.spend-calls {
  margin-left: auto;
  font-size: 10px;
  color: #999;
}

.limit-chip {
  margin-left: auto;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.05em;
  color: #FFF;
  background: #C62828;
  padding: 2px 6px;
  border-radius: 2px;
}

.mono { font-family: 'JetBrains Mono', monospace; }
</style>
