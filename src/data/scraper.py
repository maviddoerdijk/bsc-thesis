import requests
import time
from bs4 import BeautifulSoup

def fetch_live_etf_tickers():
    base_url = "https://finance.yahoo.com/research-hub/screener/etf/?start={}&count=25"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/122.0.0.0 Safari/537.36'
    }

    all_tickers = set()

    for start in range(0, 875, 25):  # 0 to 850 (inclusive) - in this case we know the webpage shows a total of 871 ETFs. 
        url = base_url.format(start)
        print(f"Fetching: {url}")
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch page {start}: {response.status_code}")
            continue
        current_tickers = set() # check amount of tickers per page

        soup = BeautifulSoup(response.text, 'html.parser')
        for a_tag in soup.find_all('a', {'data-testid': 'table-cell-ticker'}):
            href = a_tag.get('href', '')
            if href.startswith('/quote/') and href.endswith('/'):
                ticker = href.split('/')[2]
                all_tickers.add(ticker)
                current_tickers.add(ticker)
        print(f"Found {len(current_tickers)} tickers on page {start}")

        time.sleep(1)  #  wait a bit to avoid rate-limiting - this worked well enough

    return sorted(all_tickers)

def load_cached_etf_tickers():
  return list(set(['AADR', 'AAPB', 'AAVM', 'AAXJ', 'ABCS', 'ABIG', 'ACWI', 'ACWX', 'ADBG', 'AFSC', 'AGIX', 'AGMI', 'AGNG', 'AGZD', 'AIA', 'AIPI', 'AIQ', 'AIRL', 'AIRR', 'ALIL', 'ALLW', 'AMDD', 'AMDG', 'AMDL', 'AMDS', 'AMID', 'AMUU', 'AMZD', 'AMZU', 'AMZZ', 'ANGL', 'AOHY', 'AOTG', 'APED', 'AQWA', 'ARMG', 'ARVR', 'ASET', 'ASMG', 'AUMI', 'AVGB', 'AVGX', 'AVL', 'AVS', 'AVUQ', 'AVXC', 'BABX', 'BAFE', 'BBH', 'BCLO', 'BDGS', 'BEEX', 'BEEZ', 'BELT', 'BGRN', 'BGRO', 'BIB', 'BIS', 'BITS', 'BJK', 'BKCH', 'BKIV', 'BKWO', 'BLCN', 'BLCR', 'BMAX', 'BND', 'BNDW', 'BNDX', 'BOTT', 'BOTZ', 'BRHY', 'BRKD', 'BRKU', 'BRNY', 'BRRR', 'BRTR', 'BSCP', 'BSCQ', 'BSCR', 'BSCS', 'BSCT', 'BSCU', 'BSCV', 'BSCW', 'BSCX', 'BSCY', 'BSJP', 'BSJQ', 'BSJR', 'BSJS', 'BSJT', 'BSJU', 'BSJV', 'BSJW', 'BSMP', 'BSMQ', 'BSMR', 'BSMS', 'BSMT', 'BSMU', 'BSMV', 'BSMW', 'BSMY', 'BSVO', 'BTF', 'BTFX', 'BTGD', 'BUFC', 'BUFM', 'BUG', 'BULD', 'CA', 'CAFG', 'CALI', 'CANC', 'CANQ', 'CARY', 'CATH', 'CCNR', 'CCSB', 'CDC', 'CDL', 'CEFA', 'CFA', 'CFO', 'CHGX', 'CHPS', 'CIBR', 'CIL', 'CLOA', 'CLOD', 'CLOU', 'CLSM', 'CNCR', 'COIG', 'COMT', 'CONI', 'CONL', 'COPJ', 'COPP', 'CORO', 'COWG', 'COWS', 'CPLS', 'CRMG', 'CRWL', 'CSA', 'CSB', 'CTEC', 'CXSE', 'CZAR', 'DALI', 'DAPP', 'DAX', 'DECO', 'DEMZ', 'DFGP', 'DFGX', 'DGCB', 'DGRE', 'DGRS', 'DGRW', 'DIVD', 'DLLL', 'DMAT', 'DMXF', 'DRIV', 'DTCR', 'DUKH', 'DUKX', 'DVAL', 'DVOL', 'DVQQ', 'DVY', 'DWAS', 'DWSH', 'DWUS', 'DYNI', 'DYTA', 'EBI', 'ECOW', 'EEMA', 'EFAS', 'EGGQ', 'EHLS', 'EKG', 'ELFY', 'ELIL', 'ELIS', 'EMB', 'EMEQ', 'EMIF', 'EMXC', 'EMXF', 'ENDW', 'EQRR', 'ERET', 'ERNZ', 'ESGD', 'ESGE', 'ESGU', 'ESMV', 'ESPO', 'ETEC', 'ETHA', 'EUFN', 'EVMT', 'EVSD', 'EVYM', 'EWJV', 'EWZS', 'EYEG', 'FAAR', 'FAB', 'FAD', 'FALN', 'FBL', 'FBOT', 'FBZ', 'FCA', 'FCAL', 'FCEF', 'FCTE', 'FDCF', 'FDFF', 'FDIG', 'FDIV', 'FDNI', 'FDT', 'FDTX', 'FEAT', 'FEM', 'FEMB', 'FEMS', 'FEP', 'FEPI', 'FEUZ', 'FEX', 'FGM', 'FICS', 'FID', 'FINE', 'FINX', 'FIVY', 'FIXD', 'FJP', 'FKU', 'FLDB', 'FLN', 'FMB', 'FMED', 'FMHI', 'FMTM', 'FMUB', 'FMUN', 'FNK', 'FNX', 'FNY', 'FPA', 'FPXE', 'FPXI', 'FSCS', 'FSZ', 'FTA', 'FTAG', 'FTC', 'FTCS', 'FTDS', 'FTGC', 'FTGS', 'FTHI', 'FTQI', 'FTRI', 'FTSL', 'FTSM', 'FTXG', 'FTXH', 'FTXL', 'FTXN', 'FTXO', 'FV', 'FVC', 'FYC', 'FYT', 'FYX', 'GBUG', 'GFLW', 'GGLL', 'GGLS', 'GIND', 'GLCR', 'GLDI', 'GLDY', 'GLOW', 'GNMA', 'GNOM', 'GOVI', 'GPIQ', 'GPIX', 'GQQQ', 'GRID', 'GSIB', 'GTR', 'GXDW', 'HCOW', 'HECO', 'HEQQ', 'HERD', 'HERO', 'HFSP', 'HIDE', 'HIMZ', 'HISF', 'HLAL', 'HNDL', 'HOOG', 'HOOX', 'HRTS', 'HWAY', 'HYBI', 'HYDR', 'HYLS', 'HYXF', 'HYZD', 'IBAT', 'IBB', 'IBBQ', 'IBGA', 'IBGB', 'IBGK', 'IBGL', 'IBIT', 'IBOT', 'IBTF', 'IBTG', 'IBTH', 'IBTI', 'IBTJ', 'IBTK', 'IBTL', 'IBTM', 'IBTO', 'IBTP', 'IBTQ', 'ICLN', 'ICOP', 'IEF', 'IEI', 'IEUS', 'IFGL', 'IFV', 'IGF', 'IGIB', 'IGOV', 'IGSB', 'IHYF', 'IJT', 'ILIT', 'IMCV', 'IMOM', 'INDH', 'INDY', 'INFR', 'INRO', 'INTW', 'IONL', 'IONX', 'IPKW', 'IQQQ', 'ISHG', 'ISHP', 'ISTB', 'IUS', 'IUSB', 'IUSG', 'IUSV', 'IVAL', 'IVEG', 'IWTR', 'IXUS', 'JDOC', 'JEPQ', 'JGLO', 'JIVE', 'JMID', 'JPEF', 'JPY', 'JSMD', 'JSML', 'JTEK', 'KBAB', 'KBWB', 'KBWD', 'KBWP', 'KBWR', 'KBWY', 'KNGZ', 'KPDD', 'KQQQ', 'KRMA', 'KROP', 'LAYS', 'LDEM', 'LDSF', 'LEGR', 'LEXI', 'LFSC', 'LGCF', 'LGRO', 'LITP', 'LIVR', 'LMBS', 'LRGE', 'LRND', 'LVHD', 'MAXI', 'MBB', 'MBS', 'MCDS', 'MCHI', 'MCHS', 'MCSE', 'MDIV', 'MEDX', 'MEMS', 'METD', 'MFLX', 'MILN', 'MKAM', 'MNTL', 'MODL', 'MOOD', 'MQQQ', 'MRAL', 'MSFD', 'MSFL', 'MSFU', 'MSTX', 'MUD', 'MULL', 'MUU', 'MVLL', 'MYCF', 'MYCG', 'MYCH', 'MYCI', 'MYCJ', 'MYCK', 'MYCL', 'MYCM', 'MYCN', 'MYMG', 'MYMH', 'MYMI', 'MYMJ', 'NATO', 'NCIQ', 'NCPB', 'NERD', 'NEWZ', 'NFTY', 'NFXS', 'NIKL', 'NIXT', 'NPFI', 'NSI', 'NUSB', 'NVDD', 'NVDG', 'NVDL', 'NVDS', 'NVDU', 'NXTG', 'NZAC', 'NZUS', 'OBIL', 'ODDS', 'ONEQ', 'OOQB', 'OOSB', 'OPTZ', 'ORCX', 'ORR', 'OZEM', 'PABD', 'PABU', 'PALU', 'PANG', 'PATN', 'PBQQ', 'PCMM', 'PDBA', 'PDBC', 'PDP', 'PEPS', 'PEY', 'PEZ', 'PFF', 'PFM', 'PGJ', 'PHO', 'PID', 'PIE', 'PIO', 'PIZ', 'PKW', 'PLTD', 'PLTU', 'PMBS', 'PNQI', 'PPH', 'PPI', 'PQAP', 'PQJA', 'PQJL', 'PQOC', 'PRFZ', 'PRN', 'PSC', 'PSCC', 'PSCD', 'PSCE', 'PSCF', 'PSCH', 'PSCI', 'PSCM', 'PSCT', 'PSCU', 'PSET', 'PSL', 'PSTR', 'PSWD', 'PTF', 'PTH', 'PTIR', 'PTNQ', 'PUI', 'PXI', 'PY', 'PYPG', 'PYZ', 'QABA', 'QBIG', 'QBUF', 'QCLN', 'QCLR', 'QCML', 'QDTY', 'QHDG', 'QMOM', 'QNXT', 'QOWZ', 'QQA', 'QQEW', 'QQJG', 'QQLV', 'QQQ', 'QQQA', 'QQQE', 'QQQG', 'QQQH', 'QQQI', 'QQQJ', 'QQQM', 'QQQP', 'QQQS', 'QQQT', 'QQQY', 'QQXT', 'QRMI', 'QSIX', 'QSML', 'QTEC', 'QTOP', 'QTR', 'QTUM', 'QVAL', 'QYLD', 'QYLG', 'RAA', 'RAYS', 'RDTL', 'RDTY', 'RDVY', 'REAI', 'REIT', 'RFDI', 'RFEU', 'RGTX', 'RING', 'RKLX', 'RNEM', 'RNEW', 'RNRG', 'ROBT', 'ROE', 'RTH', 'RUNN', 'SARK', 'SCZ', 'SDG', 'SDTY', 'SDVY', 'SEEM', 'SEIE', 'SEIS', 'SFLO', 'SHRY', 'SHV', 'SHY', 'SIXG', 'SKOR', 'SKRE', 'SKYU', 'SKYY', 'SLQD', 'SLVO', 'SLVR', 'SMCF', 'SMCL', 'SMCO', 'SMCX', 'SMCZ', 'SMH', 'SMRI', 'SMST', 'SNSR', 'SOCL', 'SOFX', 'SOLT', 'SOLZ', 'SOXQ', 'SOXX', 'SPAM', 'SPAQ', 'SPCX', 'SPCY', 'SPRX', 'SPYQ', 'SQLV', 'SQQQ', 'SRET', 'STNC', 'SUSB', 'SUSC', 'SUSL', 'TARK', 'TAX', 'TAXE', 'TBIL', 'TCHI', 'TDI', 'TDIV', 'TDSC', 'TEKX', 'TEKY', 'THMZ', 'TLT', 'TMET', 'TPLS', 'TQQQ', 'TQQY', 'TSEL', 'TSL', 'TSLG', 'TSLL', 'TSLQ', 'TSLR', 'TSLS', 'TSMG', 'TSMU', 'TSMX', 'TSMZ', 'TSPY', 'TSYY', 'TTEQ', 'TUG', 'TUGN', 'TUR', 'TXSS', 'TXUE', 'TXUG', 'UAE', 'UBND', 'UBRL', 'UCYB', 'UEVM', 'UFIV', 'UFO', 'UITB', 'UIVM', 'ULVM', 'UMMA', 'UPGR', 'URNJ', 'USAF', 'USCL', 'USDX', 'USIG', 'USMC', 'USOI', 'USRD', 'USSH', 'USTB', 'USVM', 'USVN', 'USXF', 'UTEN', 'UTHY', 'UTRE', 'UTWO', 'UTWY', 'UYLD', 'VBIL', 'VCIT', 'VCLT', 'VCRB', 'VCSH', 'VFLO', 'VGIT', 'VGLT', 'VGSH', 'VGSR', 'VGUS', 'VIGI', 'VMBS', 'VNQI', 'VOLT', 'VONE', 'VONG', 'VONV', 'VPLS', 'VRIG', 'VRTL', 'VSDA', 'VSMV', 'VTC', 'VTHR', 'VTIP', 'VTWG', 'VTWO', 'VTWV', 'VWOB', 'VXUS', 'VYMI', 'WABF', 'WBND', 'WCBR', 'WCLD', 'WEEI', 'WGMI', 'WINC', 'WISE', 'WNDY', 'WOOD', 'WRND', 'WTBN', 'WTMU', 'WTMY', 'XAIX', 'XBIL', 'XCNY', 'XFIX', 'XOVR', 'XT', 'XYZG', 'YLDE', 'YOKE', 'YQQQ', 'YSPY', 'ZAP', 'ZIPP', 'ZTEN', 'ZTOP', 'ZTRE', 'ZTWO']))