const PLANE =
  "M0,-1 L.13,-.35 L1,.12 L1,.3 L.13,.12 L.1,.68 L.36,.88 L.36,1 L0,.92 L-.36,1 L-.36,.88 L-.1,.68 L-.13,.12 L-1,.3 L-1,.12 L-.13,-.35Z";

/** Legend for everything on the scope, the data tags and the units. Toggled with K or the Key button. */
export default function KeyPanel({ onClose }: { onClose: () => void }) {
  return (
    <div className="keypanel" role="dialog" aria-label="Key">
      <div className="kh"><b>Key</b><button onClick={onClose} aria-label="Close key">×</button></div>
      <div className="kgrid">
        <div>
          <h5>Who the question is about</h5>
          <p><svg width="22" height="18"><circle cx="11" cy="9" r="7" fill="none" stroke="var(--veh)" strokeWidth="2" /></svg>Yellow ring: the <b>crossing vehicle</b> you are clearing or holding</p>
          <p><svg width="22" height="18"><circle cx="11" cy="9" r="7" fill="none" stroke="var(--alert)" strokeWidth="2" /></svg>Red ring: the <b>runway traffic</b> it has to stay clear of</p>
          <p>Everything else is dimmed. Click a target or a strip to highlight it.</p>
          <h5>On the scope</h5>
          <p><svg width="22" height="18" viewBox="-1.2 -1.2 2.4 2.4"><path d={PLANE} fill="currentColor" /></svg>Aircraft, pointing where it is heading</p>
          <p><svg width="22" height="18"><rect x="6" y="4" width="10" height="10" transform="rotate(45 11 9)" fill="var(--veh)" /></svg>Ground vehicle (rescue, ops, maintenance)</p>
          <p><svg width="22" height="18"><circle cx="5" cy="9" r="1.5" fill="var(--tag)" opacity=".6" /><circle cx="10" cy="9" r="1.5" fill="var(--tag)" opacity=".4" /><circle cx="15" cy="9" r="1.5" fill="var(--tag)" opacity=".25" /></svg>Where it was over the last 5 seconds</p>
          <p><svg width="22" height="18"><line x1="2" y1="9" x2="20" y2="9" stroke="var(--tag)" /></svg>Where it will be in 10 s at its current speed</p>
          <p><svg width="22" height="18"><rect x="3" y="3" width="16" height="12" fill="var(--alert)" fillOpacity=".15" stroke="var(--alert)" strokeDasharray="3 2" /></svg>Crossing zone on the runway</p>
          <p><svg width="22" height="18"><line x1="3" y1="9" x2="19" y2="9" stroke="var(--alert)" strokeWidth="3" /></svg>Stop bar: red until you clear the crossing</p>
          <p><svg width="22" height="18"><line x1="3" y1="9" x2="19" y2="9" stroke="var(--ok)" strokeWidth="2" strokeDasharray="3 2" /></svg>Cleared: path across the runway</p>
          <p><svg width="22" height="18"><circle cx="11" cy="9" r="6" fill="none" stroke="var(--ai)" strokeDasharray="3 2" /></svg>AI forecast: aircraft path and its uncertainty</p>
          <p><svg width="22" height="18"><circle cx="11" cy="9" r="6" fill="none" stroke="currentColor" strokeDasharray="2 2" /></svg>Lost track: last known position</p>
        </div>
        <div>
          <h5>Data tag</h5>
          <p className="mono">AC5849<br />ARR 141 kt LOST 7s</p>
          <p>Line 1 is the ID (AC = aircraft, VH = vehicle). Line 2 is what it is doing and its speed.</p>
          <p><b>ARR</b> landing · <b>DEP</b> taking off · <b>RWY</b> fast on a runway · <b>TAXI</b> on taxiways · <b>VEH</b> vehicle</p>
          <p><b>OLD 3s</b> last report is 3 s old · <b>LOST 7s</b> no report for 7 s</p>
          <h5>Units</h5>
          <p><b>kt</b> knots, the aviation unit of speed. 1 kt = 1.15 mph = 1.85 km/h. A landing jet does about 140 kt (160 mph); a taxiing one about 15 kt.</p>
          <p><b>ft</b> feet (1 ft = 0.3 m) · <b>NM</b> nautical miles (1 NM = 1.85 km) · <b>Z / UTC</b> universal time</p>
          <h5>Strips and scoring</h5>
          <p><b>X</b> you cleared it to cross · <b>HS</b> you told it to hold short · <b>XING</b> crossing</p>
          <p><b>Brier score</b>: how far your confidence was from what happened. 0 is perfect, 1 is confidently wrong. Points = 100 × (1 − Brier).</p>
          <h5>Controls</h5>
          <p><b>C</b>/<b>H</b> call · <b>1</b>/<b>2</b>/<b>3</b> confidence · <b>Enter</b> transmit / continue · <b>Space</b> skip · scroll or <b>+</b>/<b>−</b> zoom · drag to pan · <b>0</b> reset view</p>
        </div>
      </div>
    </div>
  );
}
