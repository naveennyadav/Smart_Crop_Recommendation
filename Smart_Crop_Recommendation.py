"""Smart Crop Recommendation | single-file Streamlit web dashboard.
INSTALL (Windows PowerShell or terminal):
  python -m pip install "streamlit>=1.50,<2" "pandas>=2.2,<3" "plotly>=5.24,<8"
RUN:
  python -m streamlit run Smart_Crop_Recommendation.py
The app works offline after installation. No API key, database or model download.
Crop envelopes: FAO EcoCrop datasheets, checked 2026-10-08. This application uses
its own transparent screening algorithm, not an official FAO model or a trained ML model.
Annual rainfall is a climate screen, not irrigation demand. No yield/profit prediction.
"""
from __future__ import annotations
import html
import io
import json
import math
from datetime import datetime, timezone
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

VERSION = '1.0.0'
RETRIEVED = '2026-10-08'
COLORS = ['#147D64', '#2C77B8', '#C78A20', '#7B61AD', '#D05C52', '#2B9EA0', '#64748B', '#8A9B35']
TEXTURES = ['Light', 'Medium', 'Heavy', 'Organic']
DRAINAGES = ['Well drained', 'Excessively drained', 'Poorly drained']
WEIGHTS = {'Temperature': .30, 'Annual rainfall': .30, 'Soil pH': .25, 'Texture': .10, 'Drainage': .05}
# Numeric tuple order: absolute minimum, optimal minimum, optimal maximum, absolute maximum.
# Crop categories are navigation labels, not copied classifications.
PROFILES = [
 dict(name='Rice',scientific='Oryza sativa',group='Cereals',icon='🌾',id=1574,
      temp=(10,20,30,36),rain=(1000,1500,2000,4000),ph=(4.5,5.5,7,9),
      texture=TEXTURES,texture_abs=TEXTURES,drain=['Poorly drained'],drain_abs=['Poorly drained']),
 dict(name='Maize',scientific='Zea mays ssp. mays',group='Cereals',icon='🌽',id=238663,
      temp=(10,18,33,47),rain=(400,600,1200,1800),ph=(4.5,5,7,8.5),
      texture=['Medium','Organic'],texture_abs=['Heavy','Medium','Light','Organic'],
      drain=['Well drained'],drain_abs=['Well drained','Excessively drained']),
 dict(name='Wheat',scientific='Triticum aestivum',group='Cereals',icon='🌾',id=2114,
      temp=(5,15,23,27),rain=(300,750,900,1600),ph=(5.5,6,7,8.5),
      texture=['Medium','Organic'],texture_abs=['Heavy','Medium','Organic'],
      drain=['Well drained'],drain_abs=['Well drained']),
 dict(name='Sorghum',scientific='Sorghum bicolor',group='Cereals',icon='🌾',id=48747,
      temp=(8,22,35,40),rain=(300,400,600,700),ph=(5,5.5,7.5,8),
      texture=['Heavy','Medium'],texture_abs=TEXTURES,
      drain=['Well drained'],drain_abs=['Well drained','Excessively drained']),
 dict(name='Pearl millet',scientific='Pennisetum glaucum',group='Cereals',icon='🌿',id=8418,
      temp=(12,25,35,40),rain=(200,400,900,1700),ph=(4.5,5,6.5,8.3),
      texture=['Medium','Light'],texture_abs=['Medium','Light'],
      drain=['Well drained'],drain_abs=['Well drained','Excessively drained']),
 dict(name='Groundnut',scientific='Arachis hypogaea',group='Oilseeds',icon='🥜',id=2199,
      temp=(10,22,32,45),rain=(400,600,1500,4000),ph=(4.5,5.5,6.5,8.5),
      texture=['Medium'],texture_abs=TEXTURES,drain=['Well drained'],drain_abs=['Well drained']),
 dict(name='Sunflower',scientific='Helianthus annuus',group='Oilseeds',icon='🌻',id=1191,
      temp=(5,17,34,45),rain=(300,600,1000,1600),ph=(5.5,6,7.5,8),
      texture=['Medium','Light'],texture_abs=['Heavy','Medium','Light'],
      drain=['Well drained'],drain_abs=['Well drained']),
 dict(name='Chickpea',scientific='Cicer arietinum',group='Pulses',icon='🌱',id=2479,
      temp=(7,15,29,35),rain=(300,600,1000,1800),ph=(4.7,6,8.5,9.5),
      texture=['Heavy','Medium'],texture_abs=['Heavy','Medium','Light'],
      drain=['Well drained'],drain_abs=['Well drained','Excessively drained']),
]
BY_NAME = {p['name']:p for p in PROFILES}
PRESETS = {
 'Dryland example': dict(field_name='Dryland demonstration',temp_c=29.,rain_year_mm=550.,ph=6.8,texture='Medium',drainage='Well drained',origin='Demonstration inputs'),
 'Wetland example': dict(field_name='Wetland demonstration',temp_c=27.,rain_year_mm=1700.,ph=6.2,texture='Heavy',drainage='Poorly drained',origin='Demonstration inputs'),
 'Cool climate example': dict(field_name='Cool climate demonstration',temp_c=20.,rain_year_mm=800.,ph=6.4,texture='Medium',drainage='Well drained',origin='Demonstration inputs'),
}
FACTORS = list(WEIGHTS)
CSV_COLUMNS = ['field_name','temp_c','rain_year_mm','ph','texture','drainage']

def source(p):
    return f"https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id={p['id']}"

def validate(field):
    errors=[]
    for key,low,high in [('temp_c',-20,60),('rain_year_mm',0,12000),('ph',0,14)]:
        try:
            val=float(field[key])
            if not math.isfinite(val) or not low <= val <= high:
                errors.append(f'{key} must be finite and between {low} and {high}.')
        except (ValueError,TypeError,KeyError): errors.append(f'{key} must be numeric.')
    if field.get('texture') not in TEXTURES: errors.append('Unknown texture. Use Light, Medium, Heavy or Organic.')
    if field.get('drainage') not in DRAINAGES: errors.append('Unknown drainage category.')
    name=field.get('field_name')
    if not isinstance(name,str) or not name.strip() or len(name)>100: errors.append('field_name must contain 1–100 characters.')
    return errors

def range_score(value, bounds):
    """100 throughout optimum, linearly taper to 20 at limits, 0 outside."""
    a,lo,hi,b=bounds
    if value<a or value>b: return 0.
    if lo<=value<=hi: return 100.
    if value<lo: return 20.+80.*(value-a)/(lo-a)
    return 20.+80.*(b-value)/(b-hi)

def category_score(value, preferred, tolerated):
    return 100. if value in preferred else (60. if value in tolerated else 0.)

def grade(score):
    return 'Strong match' if score>=80 else 'Moderate match' if score>=60 else 'Limited match' if score>0 else 'Outside profile'

def evaluate(field):
    errors=validate(field)
    if errors: raise ValueError(' '.join(errors))
    rows=[]
    for p in PROFILES:
        scores={'Temperature':range_score(float(field['temp_c']),p['temp']),
                'Annual rainfall':range_score(float(field['rain_year_mm']),p['rain']),
                'Soil pH':range_score(float(field['ph']),p['ph']),
                'Texture':category_score(field['texture'],p['texture'],p['texture_abs']),
                'Drainage':category_score(field['drainage'],p['drain'],p['drain_abs'])}
        blockers=[k for k,v in scores.items() if v==0]
        value=0. if blockers else sum(scores[k]*WEIGHTS[k] for k in FACTORS)
        rows.append({'Crop':p['name'],'Group':p['group'],'Score':round(value,2),'Match':grade(round(value,2)),
                     'Limiting factor':', '.join(blockers) if blockers else min(scores,key=scores.get),
                     'Outside factors':', '.join(blockers),**{k:round(v,2) for k,v in scores.items()},'Source':source(p)})
    return pd.DataFrame(rows).sort_values(['Score','Crop'],ascending=[False,True],ignore_index=True)

def safe_csv(frame):
    copy=frame.copy()
    # Avoid interpreting uploaded field names as spreadsheet formulas.
    for c in copy.select_dtypes(include=['object','string']).columns:
        copy[c]=copy[c].map(lambda v: "'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@','\t','\r')) else v)
    return copy.to_csv(index=False).encode('utf-8-sig')

def batch_evaluate(frame):
    missing=set(CSV_COLUMNS)-set(frame.columns)
    if missing: raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    if not 1<=len(frame)<=2000: raise ValueError('Upload between 1 and 2,000 data rows.')
    records=[]
    for i,(_,row) in enumerate(frame.iterrows(),start=2):
        f={k:row[k] for k in CSV_COLUMNS}
        errors=validate(f)
        if errors:
            records.append({'CSV row':i,'Field':str(f.get('field_name','')),'Status':'Invalid','Top crop':'','Score':None,'Tied crops':'','Notes':' '.join(errors)})
        else:
            r=evaluate(f); top=r.iloc[0]; tied=r.loc[r.Score.eq(top.Score),'Crop'].tolist() if top.Score>0 else []
            records.append({'CSV row':i,'Field':f['field_name'],'Status':'Screened' if top.Score>0 else 'No match',
                            'Top crop':top.Crop if top.Score>0 else '', 'Score':top.Score,
                            'Tied crops':', '.join(tied) if len(tied)>1 else '',
                            'Notes':top['Match'] if top.Score>0 else 'All eight crops are outside at least one profile constraint.'})
    return pd.DataFrame(records)

def printable_report(f, ranking):
    esc=lambda v: html.escape(str(v))
    metadata=''.join(f'<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>' for k,v in f.items())
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Smart Crop Recommendation report</title>
<style>body{{font:16px/1.6 Arial,sans-serif;color:#18382c;max-width:1000px;margin:40px auto;padding:20px}}h1{{color:#147d64}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}}td,th{{border:1px solid #dce6e0;padding:9px;text-align:left}}th{{background:#edf5f0}}a{{color:#147d64}}@media print{{body{{margin:0}}table{{page-break-inside:auto}}tr{{break-inside:avoid}}}}</style>
<h1>Smart Crop Recommendation</h1><p>Field screening report · {esc(datetime.now(timezone.utc).isoformat(timespec='seconds'))} UTC</p>
<table>{metadata}</table><h2>Crop rankings</h2>{ranking.to_html(index=False,escape=True)}
<h2>Method and interpretation</h2><p>Scores are screening indices out of 100, not probabilities or yield forecasts. Optimal numeric ranges score 100; tolerance edges score 20; outside ranges score 0. Weights: temperature 30%, annual rainfall 30%, soil pH 25%, texture 10%, drainage 5%. Any zero factor makes the overall score zero. Equal scores are tied; alphabetical order is only a display choice.</p>
<p>Annual rainfall is the site's annual climate total. It is not crop-season water demand or a substitute for an irrigation plan. This screen omits cultivar, rainfall timing, nutrients, salinity, pests and economics. Verify local growing conditions and variety recommendations before planting.</p>
<p>Sources: FAO EcoCrop crop datasheets (links in the table), checked {RETRIEVED}. The application uses an independently designed rule engine, not an official FAO model. Version {VERSION}.</p></html>'''

def plot(fig,key,height=390):
    fig.update_layout(paper_bgcolor='white',plot_bgcolor='white',height=height,font=dict(family='Segoe UI, Arial',size=13,color='#203D33'),
                      margin=dict(l=25,r=25,t=55,b=55),colorway=COLORS,title_font=dict(size=17),
                      legend=dict(orientation='h',y=-.2,x=0,title_text=''),hoverlabel=dict(bgcolor='#173B2C',font_color='white'))
    fig.update_xaxes(gridcolor='#EDF2EE');fig.update_yaxes(gridcolor='#EDF2EE')
    st.plotly_chart(fig,key="chart_"+key,width='stretch',theme=None,config={'displaylogo':False,'toImageButtonOptions':{'format':'png','scale':2}})

def heading(title,detail=''):
    st.markdown(f'<h2 class="section">{html.escape(title)}</h2><p class="muted">{html.escape(detail)}</p>',unsafe_allow_html=True)

def kpis(items):
    for col,(label,value,note) in zip(st.columns(len(items)),items):
        with col: st.markdown(f'<div class="kpi"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(str(value))}</div><div class="muted">{html.escape(note)}</div></div>',unsafe_allow_html=True)

def set_preset(name):
    f=PRESETS[name].copy();st.session_state['field']=f
    for k,v in f.items(): st.session_state['input_'+k]=v

CSS='''<style>
.stApp{background:#F4F7F2;color:#18382C}h1,h2,h3{color:#173B2C!important}p,li{line-height:1.65}.block-container{max-width:1450px;padding-top:1.8rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#102F25;border-right:1px solid #284B3E}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#EEF6F0!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div,[data-testid="stSidebar"] input{background:#fff!important;color:#173B2C!important;-webkit-text-fill-color:#173B2C!important}
[data-testid="stSidebar"] [data-baseweb="select"] span{color:#173B2C!important}
[data-testid="stSidebar"] button{background:#E5F2E8;color:#173B2C;border:1px solid #AAC9B6}
.hero{padding:32px 36px;border-radius:22px;background:linear-gradient(120deg,#123D2D,#226B50);color:white;position:relative;overflow:hidden;margin-bottom:20px;border:1px solid #31785B}
.hero:after{content:"";position:absolute;width:360px;height:360px;border:40px solid #ffffff0b;border-radius:50%;right:-50px;top:-220px;pointer-events:none}
.hero h1{font-size:38px!important;color:#fff!important;letter-spacing:-1px;margin:8px 0}.hero p{color:#D5EADC!important;max-width:720px;margin:8px 0}.eyebrow{color:#BAE3BC;font-size:11px;font-weight:700;letter-spacing:2px}.pill{display:inline-block;padding:6px 12px;border:1px solid #ffffff33;border-radius:100px;color:#E6F4E9;font-size:11px;margin-top:12px}
.kpi{background:white;border:1px solid #DAE5DC;border-radius:16px;padding:19px;min-height:125px;margin-bottom:18px;border-top:3px solid #58A783}.label{text-transform:uppercase;letter-spacing:1px;font-size:11px;color:#6B8176}.value{font-weight:800;font-size:28px;margin:5px 0;color:#153D2B}.muted{font-size:13px;color:#677C70}.section{font-size:24px!important;margin:14px 0 2px!important}
.crop-card{background:white;border:1px solid #DCE7DE;border-radius:18px;padding:24px;min-height:222px;margin-bottom:18px}.crop-card.first{background:#E8F3E7;border-color:#B4D2AC}.crop-card h3{margin:8px 0;font-size:24px}.score{font-size:36px;font-weight:800;color:#157A58}.crop-icon{font-size:30px}.tag{font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#5B7565}
[data-testid="stMetric"]{background:#fff;border:1px solid #DAE5DC;border-radius:14px;padding:16px}[data-testid="stMetric"] label,[data-testid="stMetricValue"]{color:#18382C!important}
[data-testid="stPlotlyChart"]{background:white;border:1px solid #DAE5DC;border-radius:16px;overflow:hidden}
[data-testid="stDataFrame"]{border:1px solid #DAE5DC;border-radius:12px}.stTabs [data-baseweb="tab-list"]{gap:8px}.stTabs [data-baseweb="tab"]{background:white;border:1px solid #DAE5DC;border-radius:8px;padding:10px 16px}
@media(max-width:750px){.hero{padding:22px}.hero h1{font-size:28px!important}.value{font-size:23px}.block-container{padding-top:1rem}}
</style>'''

def main():
    st.set_page_config(page_title='Smart Crop Recommendation',page_icon='🌱',layout='wide',initial_sidebar_state='expanded')
    st.markdown(CSS,unsafe_allow_html=True)
    if 'field' not in st.session_state: set_preset('Dryland example')
    if 'snapshots' not in st.session_state: st.session_state.snapshots=[]
    with st.sidebar:
        st.markdown('## 🌱 Smart Crop')
        st.caption('SOIL • CLIMATE • CROP SUITABILITY')
        page=st.radio('Workspace',['Overview','Crop analysis','Scenario lab','Crop library','Batch screening','Reports & method'],key='nav')
        st.divider()
        example=st.selectbox('Example conditions',list(PRESETS),key='preset')
        st.button('Load example',on_click=set_preset,args=(example,),width='stretch')
        st.markdown('### Field conditions')
        st.caption('Edit the fields, then select Analyze field.')
        with st.form('field_form'):
            field_name=st.text_input('Field name',key='input_field_name',max_chars=100)
            origin=st.selectbox('Input source',['Demonstration inputs','User-supplied inputs'],key='input_origin')
            temp=st.number_input('Growing-season mean temperature (°C)',min_value=-20.,max_value=60.,step=.5,key='input_temp_c',help='A representative mean for your intended growing season, not today’s maximum.')
            rain=st.number_input('Annual rainfall (mm/year)',min_value=0.,max_value=12000.,step=50.,key='input_rain_year_mm',help='Long-term annual climate total. Do not enter only one month or irrigation volume.')
            ph=st.number_input('Soil pH',min_value=0.,max_value=14.,step=.1,key='input_ph')
            texture=st.selectbox('Soil texture class',TEXTURES,key='input_texture',help='Broad screening classes: light/sandy, medium/loamy, heavy/clayey, organic. A soil report gives a better classification.')
            drainage=st.selectbox('Drainage',DRAINAGES,key='input_drainage')
            submitted=st.form_submit_button('Analyze field →',width='stretch',type='primary')
        if submitted:
            candidate=dict(field_name=field_name.strip(),temp_c=temp,rain_year_mm=rain,ph=ph,texture=texture,drainage=drainage,origin=origin)
            errors=validate(candidate)
            if errors: st.error(' '.join(errors))
            else: st.session_state.field=candidate
        st.caption('Rainfall uses annual totals. All results follow the last submitted field profile.')
    field=st.session_state.field.copy();ranking=evaluate(field)
    st.markdown('<div class="hero"><div class="eyebrow">AGRICULTURE / FIELD PLANNING</div><h1>Smart Crop Recommendation</h1><p>Understand your field. Compare crop options. See the soil and climate factors behind every recommendation.</p><span class="pill">8 crop profiles &nbsp; • &nbsp; Explainable screening &nbsp; • &nbsp; Works offline</span></div>',unsafe_allow_html=True)
    st.caption(f"ACTIVE FIELD: {field['field_name']}  •  {field['origin']}  •  {field['temp_c']:g} °C  •  {field['rain_year_mm']:g} mm/year  •  pH {field['ph']:g}")
    if field['origin']=='Demonstration inputs': st.info('Example conditions are loaded. Enter your field measurements in the sidebar and select Analyze field to personalize the results.')
    positive=ranking[ranking.Score>0]
    ties=ranking.loc[ranking.Score.eq(ranking.iloc[0].Score),'Crop'].tolist() if len(positive) else []
    if page=='Overview':
        kpis([('Profiles screened',8,'Cereals, oilseeds and pulses'),('Strong matches',int((ranking.Score>=80).sum()),'Screening index ≥80'),('Within profiles',len(positive),'No factor outside its limits'),('Best screening index',f'{ranking.iloc[0].Score:.1f}/100','Not a yield or success probability')])
        heading('Your crop shortlist','Rankings reflect the submitted field profile. Equal scores share the same rank.')
        if not len(positive): st.warning('No crop matches all profile constraints. Review your inputs and the limiting factors below; the app will not recommend an unsuitable top crop.')
        else:
            if len(ties)>1: st.caption('Joint highest score: '+', '.join(ties)+'. Display order within a tie is alphabetical.')
            for col,(_,row) in zip(st.columns(3),positive.head(3).iterrows()):
                p=BY_NAME[row.Crop]
                with col: st.markdown(f'<div class="crop-card {"first" if row.Crop in ties else ""}"><span class="tag">{p["group"]}</span><div class="crop-icon">{p["icon"]}</div><h3>{row.Crop}</h3><div class="score">{row.Score:.1f}<span style="font-size:14px"> /100</span></div><div class="muted">{row.Match} · Lowest factor: {row["Limiting factor"]}</div></div>',unsafe_allow_html=True)
        left,right=st.columns([1.35,1])
        with left:
            fig=px.bar(ranking.sort_values('Score'),x='Score',y='Crop',orientation='h',color='Score',color_continuous_scale=['#DDE8DE','#147D64'],text='Score',title='Crop suitability ranking',range_x=[0,105]);fig.update_coloraxes(showscale=False);fig.update_traces(texttemplate='%{text:.1f}',textposition='outside');plot(fig,'overview_ranking')
        with right:
            counts=ranking.Match.value_counts().reindex(['Strong match','Moderate match','Limited match','Outside profile'],fill_value=0)
            fig=go.Figure(go.Pie(labels=counts.index,values=counts.values,hole=.73,marker_colors=['#147D64','#C78A20','#D48C59','#9EAFA5'],textinfo='value',sort=False));fig.update_layout(title='Screening outcome',annotations=[dict(text='<b>8</b><br>crops',x=.5,y=.5,showarrow=False,font_size=22)]);plot(fig,'overview_donut')
        heading('What drives the result','Zero means a factor falls outside the stored profile. It blocks the overall recommendation.')
        fig=go.Figure(go.Heatmap(z=ranking[FACTORS].values,x=FACTORS,y=ranking.Crop,zmin=0,zmax=100,colorscale=[[0,'#F7DFD8'],[.6,'#F6EBC9'],[1,'#6DB58C']],text=ranking[FACTORS].round().values,texttemplate='%{text:.0f}',textfont_color='#173B2C',hovertemplate='%{y}<br>%{x}: %{z:.1f}/100<extra></extra>',colorbar_title='Index'));fig.update_yaxes(autorange='reversed');plot(fig,'overview_heat',420)
        st.caption('This is an initial climate-and-soil screen. Confirm cultivar, planting season, rainfall distribution, soil nutrients and local recommendations before planting.')
    elif page=='Crop analysis':
        heading('Compare the details','Inspect factor scores and identify what prevents a crop from matching your field.')
        choices=st.multiselect('Compare up to four crops',ranking.Crop.tolist(),default=ranking.Crop.head(3).tolist(),max_selections=4,key='compare')
        if choices:
            selected=ranking.set_index('Crop').loc[choices].reset_index()
            a,b=st.columns([1,1.1])
            with a:
                fig=go.Figure()
                for name in choices:
                    vals=ranking.set_index('Crop').loc[name,FACTORS].astype(float).tolist();fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=FACTORS+[FACTORS[0]],fill='toself',name=name,opacity=.65))
                fig.update_layout(title='Factor comparison',polar=dict(radialaxis=dict(visible=True,range=[0,100]),bgcolor='white'));plot(fig,'compare_radar',430)
            with b: st.dataframe(selected[['Crop','Score','Match','Outside factors']].rename(columns={'Score':'Index /100'}),hide_index=True,width='stretch')
        else: st.info('Select at least one crop to compare.')
        name=st.selectbox('Inspect a crop',ranking.Crop.tolist(),key='inspect');p=BY_NAME[name];row=ranking.set_index('Crop').loc[name]
        if row.Score==0: st.warning('Outside this stored profile: '+row['Outside factors']+'. Zero is a screening exclusion, not proof that every cultivar will fail.')
        else: st.success(f'{name}: {row.Score:.1f}/100 · {row.Match}')
        observations=[]
        for label,key,input_key,unit in [('Temperature','temp','temp_c','°C'),('Annual rainfall','rain','rain_year_mm','mm/year'),('Soil pH','ph','ph','')]:
            aa,lo,hi,bb=p[key];v=field[input_key];status='Optimal band' if lo<=v<=hi else 'Tolerance band' if aa<=v<=bb else 'Outside limits'
            observations.append({'Factor':label,'Your field':f'{v:g} {unit}','Optimal':f'{lo:g}–{hi:g} {unit}','Absolute limits':f'{aa:g}–{bb:g} {unit}','Result':status})
        observations.extend([{'Factor':'Texture','Your field':field['texture'],'Optimal':', '.join(p['texture']),'Absolute limits':', '.join(p['texture_abs']),'Result':f'{row.Texture:g}/100'}, {'Factor':'Drainage','Your field':field['drainage'],'Optimal':', '.join(p['drain']),'Absolute limits':', '.join(p['drain_abs']),'Result':f'{row.Drainage:g}/100'}])
        st.dataframe(pd.DataFrame(observations),hide_index=True,width='stretch')
        st.link_button(f'View {name} source profile ↗',source(p))
    elif page=='Scenario lab':
        heading('Explore a different climate','Change hypothetical conditions without overwriting your active field. These are scenarios, not forecasts.')
        a,b,c=st.columns(3)
        with a: dt=st.slider('Temperature change (°C)',-10.,10.,0.,.5)
        with b: dr=st.slider('Annual rainfall change (%)',-80,100,0,5)
        with c: dp=st.slider('Hypothetical soil pH change',-2.,2.,0.,.1,help='A scenario only. This is not a recommendation to apply lime, acid or fertilizer.')
        scenario={**field,'temp_c':min(60.,max(-20.,field['temp_c']+dt)),'rain_year_mm':min(12000.,field['rain_year_mm']*(1+dr/100)),'ph':min(14.,max(0.,field['ph']+dp))}
        st.caption(f"Scenario: {scenario['temp_c']:g} °C · {scenario['rain_year_mm']:g} mm/year · pH {scenario['ph']:g}. Values stay within the input limits.")
        future=evaluate(scenario);merged=ranking[['Crop','Score']].merge(future[['Crop','Score']],on='Crop',suffixes=(' baseline',' scenario'));merged['Change']=merged['Score scenario']-merged['Score baseline']
        fig=px.bar(merged.melt(id_vars=['Crop','Change'],value_vars=['Score baseline','Score scenario'],var_name='Conditions',value_name='Index'),x='Crop',y='Index',color='Conditions',barmode='group',range_y=[0,105],title='Baseline vs scenario');plot(fig,'scenario_bars')
        st.dataframe(merged.round(2),hide_index=True,width='stretch')
        crop=st.selectbox('Temperature sensitivity for',ranking.Crop.tolist(),key='sensitivity')
        points=[]
        for t in range(-10,51):
            value=evaluate({**field,'temp_c':float(t)}).set_index('Crop').loc[crop,'Score'];points.append({'Temperature (°C)':t,'Index':value})
        fig=px.line(pd.DataFrame(points),x='Temperature (°C)',y='Index',title=f'{crop} temperature sensitivity · other baseline inputs fixed',range_y=[0,105]);fig.add_vline(x=field['temp_c'],line_dash='dash',line_color='#C78A20');plot(fig,'sensitivity')
        st.caption('A flat zero line can be caused by another factor outside the profile, even when temperature improves.')
    elif page=='Crop library':
        heading('Crop profile library','Inspect the environmental ranges used by the screening engine. Source links are provided for every crop.')
        group=st.selectbox('Crop category',['All','Cereals','Oilseeds','Pulses']);query=st.text_input('Find a crop',placeholder='Try groundnut, rice or chickpea')
        crops=[p for p in PROFILES if (group=='All' or p['group']==group) and query.lower() in (p['name']+' '+p['scientific']).lower()]
        if not crops: st.info('No profiles match your search.')
        for p in crops:
            with st.expander(f"{p['icon']}  {p['name']}  ·  {p['scientific']}",expanded=len(crops)==1):
                rows=[]
                for label,key,unit in [('Temperature','temp','°C'),('Annual rainfall','rain','mm/year'),('Soil pH','ph','')]:
                    a,lo,hi,b=p[key];rows.append({'Factor':label,'Optimal range':f'{lo:g}–{hi:g} {unit}','Absolute range':f'{a:g}–{b:g} {unit}'})
                st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
                st.write('Preferred texture: '+', '.join(p['texture'])+' · Preferred drainage: '+', '.join(p['drain']))
                st.link_button('FAO EcoCrop datasheet ↗',source(p))
        st.caption('Source ranges describe broad species envelopes. They are not a planting calendar or cultivar-specific prescription.')
    elif page=='Batch screening':
        heading('Screen multiple fields','Upload a CSV to validate each row and compare its highest-scoring crop options. Maximum 2,000 rows / 2 MB.')
        sample=pd.DataFrame([{k:f[k] for k in CSV_COLUMNS} for f in PRESETS.values()])
        st.download_button('Download CSV template',safe_csv(sample),'crop_fields_template.csv','text/csv',key='template')
        st.caption('Required columns: '+', '.join(CSV_COLUMNS)+'. Temperature in °C; rainfall is annual mm/year. Texture and drainage must use the template categories.')
        uploaded=st.file_uploader('Field measurements CSV',type=['csv'])
        if uploaded is not None:
            try:
                if uploaded.size>2_000_000: raise ValueError('This file exceeds the 2 MB limit.')
                frame=pd.read_csv(uploaded);results=batch_evaluate(frame)
                kpis([('Fields',len(results),'Uploaded rows'),('Valid rows',int(results.Status.ne('Invalid').sum()),'Passed input validation'),('No match',int(results.Status.eq('No match').sum()),'Outside all stored profiles'),('Invalid rows',int(results.Status.eq('Invalid').sum()),'Correct the issues and upload again')])
                st.dataframe(results,hide_index=True,width='stretch')
                valid=results[results.Status.eq('Screened')]
                if len(valid):
                    counts=valid['Top crop'].value_counts().rename_axis('Crop').reset_index(name='Fields');fig=px.bar(counts,x='Crop',y='Fields',color='Crop',text='Fields',title='First-listed top crop per field');plot(fig,'batch_counts');st.caption('When crops tie, the chart uses the alphabetically first crop. See Tied crops in the table before interpreting these counts.')
                st.download_button('Download batch results',safe_csv(results),'crop_batch_results.csv','text/csv',key='batch_download')
            except (ValueError,UnicodeError,pd.errors.ParserError,pd.errors.EmptyDataError) as exc: st.error(str(exc))
        else: st.info('Download the template, replace its demonstration values with your field data, then upload it here.')
    else:
        heading('Reports and method','Export your current analysis, save comparisons during this session, and inspect how the score is calculated.')
        a,b=st.columns(2)
        export=ranking.copy()
        for k,v in field.items(): export['Input '+k]=v
        export['Engine version']=VERSION;export['Source checked']=RETRIEVED
        with a: st.download_button('Download detailed CSV',safe_csv(export),'smart_crop_recommendations.csv','text/csv',width='stretch')
        with b: st.download_button('Download printable report',printable_report(field,ranking).encode('utf-8'),'smart_crop_report.html','text/html',width='stretch')
        st.caption('Open the downloaded HTML in a browser and use Print → Save as PDF. Its charts are replaced with exact data tables.')
        if st.button('Save current field snapshot',disabled=len(st.session_state.snapshots)>=20):
            st.session_state.snapshots.append({'Saved UTC':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),**field,'Top crops':', '.join(ties) if ties else 'No match','Best index':float(ranking.iloc[0].Score)})
        if st.session_state.snapshots:
            history=pd.DataFrame(st.session_state.snapshots);st.dataframe(history,hide_index=True,width='stretch');st.download_button('Download saved snapshots',safe_csv(history),'crop_snapshots.csv','text/csv')
            if st.button('Clear saved snapshots'): st.session_state.snapshots=[];st.rerun()
        st.caption('Snapshots are held only in this browser session (maximum 20). Download them to keep a copy; they are not saved in a database.')
        with st.expander('How the recommendation score works',expanded=True):
            st.write('Each numeric factor receives 100 inside its optimal range, falls linearly to 20 at either absolute limit, and receives 0 outside. Preferred texture and drainage receive 100; tolerated categories receive 60; unsupported categories receive 0.')
            st.dataframe(pd.DataFrame({'Factor':FACTORS,'Weight (%)':[int(100*WEIGHTS[k]) for k in FACTORS]}),hide_index=True,width='stretch')
            st.write('The overall index is the weighted mean, except that any zero factor forces the overall index to zero. Strong ≥80, moderate 60–<80, limited >0–<60; zero is outside profile. The thresholds and weights are application design choices, not validated agronomic probabilities. Tied scores are listed alphabetically, without claiming one tied crop is better.')
            st.write('This independently designed rule engine is not a trained machine-learning model or the official EcoCrop model. It cannot predict yield, profit, crop failure or exact fertilizer needs. It does not fetch live weather. NPK, salinity, rainfall timing, cultivar, pests, frost events and economic conditions are outside the model.')
        with st.expander('Sources and normalization'):
            st.write('FAO EcoCrop numeric ranges were checked on '+RETRIEVED+'. Temperature is treated here as a growing-season mean screening input; rainfall follows the source’s annual basis. Confirm locally how climate summaries represent the intended season.')
            st.write('The source’s wide soil-texture category maps to all four displayed classes. If an optimal texture is absent from the source’s absolute list, this app includes it in the tolerated set so that an optimum cannot be excluded. For maize and wheat this includes Organic. Source drainage categories are shortened to Well drained, Excessively drained and Poorly drained. These categorical mappings and hard exclusions are simplifications.')
            for p in PROFILES: st.markdown(f"- [{p['name']} — {p['scientific']}]({source(p)})")
            st.write('Before field use, verify the soil test, local variety and crop calendar with an agronomist or agricultural extension service. A low annual-rainfall score does not assess the potential of an irrigated production system.')
        with st.expander('Installation and running'):
            st.code('python -m pip install "streamlit>=1.50,<2" "pandas>=2.2,<3" "plotly>=5.24,<8"\npython -m streamlit run Smart_Crop_Recommendation.py',language='powershell')
            st.write('Run these commands in the folder containing this file. Keep the terminal open and visit the Local URL displayed by Streamlit. If Windows provides the py launcher instead, substitute py for python. No API key is needed.')
    st.divider();st.caption(f'Smart Crop Recommendation · Python / Streamlit / Pandas / Plotly · v{VERSION} · Educational field screening')

if __name__=='__main__': main()
