import streamlit as st
import cohere
from streamlit_extras.switch_page_button import switch_page
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone
import random

MODEL_ID = '28b1f720-22ff-4556-85bc-2ae1e0ee18cf-ft'
COHERE_API_KEY = 'KOmzoZw2eX5jJgZBG501EbzSykrb27O0OuKFMjOj'
MOVING_AVERAGE_NUM = 3
co = cohere.Client(COHERE_API_KEY)

def getUserTimeSpent():
    """ Return the total time spent on the website in seconds. """
    return (st.session_state['current_time'] - st.session_state['start_time']).total_seconds()

def classify_session():
    product_viewd_avg_price = (st.session_state['products_viewed_sum_price']/st.session_state['products_viewed'] if st.session_state['products_viewed'] != 0 else 0)
    product_added_avg_price = (st.session_state['products_added_sum_price']/st.session_state['products_added'] if st.session_state['products_added'] != 0 else 0)
    product_removed_avg_price = (st.session_state['products_removed_sum_price']/st.session_state['products_removed'] if st.session_state['products_removed'] != 0 else 0)
    
    if response := co.classify(
        model=MODEL_ID,
        inputs=[f"{getUserTimeSpent()} {st.session_state['products_viewed']} {product_viewd_avg_price} {st.session_state['products_added']} {product_added_avg_price} {st.session_state['products_removed']} {product_removed_avg_price}"]):
        return response.classifications[0]
    return []

def removeFromCart(product_id):
    for i in range(len(user_cart)):
        if user_cart[i]['id'] == product_id:
            print(user_cart[i])
            del user_cart[i]
            st.session_state['products_removed'] += 1
        break
    return True


st.set_page_config(
    page_title="Insightly Store - Home Page",
    page_icon="✨",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------

if 'current_product' not in st.session_state:
	st.session_state['current_product'] = None
        
if 'cart' not in st.session_state:
    	st.session_state['cart'] = []
user_cart = st.session_state['cart']
current_product = st.session_state['current_product']

# ---------------------------------------------------------------------
# End Products
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# User behavior 
# ---------------------------------------------------------------------

# set variables
if 'products_viewed' not in st.session_state:
    st.session_state['products_viewed'] = 0

if 'products_viewed_hrice' not in st.session_state:
    st.session_state['products_viewed_sum_price'] = 0

if 'products_added' not in st.session_state:
    st.session_state['products_added'] = 0

if 'products_added_sum_price' not in st.session_state:
    st.session_state['products_added_sum_price'] = 0

if 'products_removed' not in st.session_state:
    st.session_state['products_removed'] = 0

if 'products_removed_sum_price' not in st.session_state:
    st.session_state['products_removed_sum_price'] = 0

if 'start_time' not in st.session_state:
    st.session_state['start_time'] = datetime.now(timezone.utc)

if 'current_time' not in st.session_state:
    st.session_state['current_time'] = datetime.now(timezone.utc)

if 'moving_average' not in st.session_state:
	st.session_state['moving_average'] = []

if 'moving_average_probability' not in st.session_state:
    	st.session_state['moving_average_probability'] = {
			'0': 0,
			'1': 0
		}

# update cureent_time 
st.session_state['current_time'] = datetime.now(timezone.utc)

session_stats = classify_session()

# update moving average
if len(st.session_state['moving_average']) >= MOVING_AVERAGE_NUM:
    	st.session_state['moving_average'].pop(0)
st.session_state['moving_average'].append([session_stats.labels['0'].confidence * 100, session_stats.labels['1'].confidence * 100])

# update moving average probability
st.session_state['moving_average_probability']['0'] = sum(
	i[0] for i in st.session_state['moving_average']
) / len(st.session_state['moving_average'])

st.session_state['moving_average_probability']['1'] = sum(
	i[1] for i in st.session_state['moving_average']
) / len(st.session_state['moving_average'])

if 'prod_msg' not in st.session_state:
    st.session_state['prod_msg'] = ''

# ---------------------------------------------------------------------
# End User behavior data
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
    [data-testid="stSidebarNav"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

# User current session details
st.sidebar.markdown(f"""
                    <center><h3>Current user behavior</h3></center>
                    Total time spent (in seconds): {str(round(getUserTimeSpent(), 2))}<br>
					Products viewed: {st.session_state['products_viewed']}<br>
					Products added: {st.session_state['products_added']}<br>
					Products removed: {st.session_state['products_removed']}<br>
                    Session stats: <br>
                    <span style="padding-left:20px;">Not purchase ({str(round(st.session_state['moving_average_probability']['0'], 4))[:5]}%)</span><br>
                    <span style="padding-left:20px;">Purchase ({str(round(st.session_state['moving_average_probability']['1'], 4))[:5]}%)</span>
					""", unsafe_allow_html=True)
# ---------------------------------------------------------------------
# End Sidebar
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------

with st.container():
    header_title, home_btn = st.columns([4,1])
    with header_title:
        st.markdown("# 🛒 Your cart  ")

    with home_btn:
        if st.button('🏠 Home', use_container_width=True):
            switch_page("main_page")
        if st.button('👥 About', use_container_width=True):
            switch_page("about")


st.write("<hr>", unsafe_allow_html=True)

if st.session_state['prod_msg'] != '':
    st.info(st.session_state['prod_msg'], icon="⚠")
    st.session_state['prod_msg'] = ''
    st_autorefresh(interval=2000, limit=2)

for product in user_cart:
    with st.container():
        R_col, l_col = st.columns((1,2))
    with R_col:
        st.image(f"{product['image']}")
    with l_col:   
        st.write(f"{product['description']}")

        with st.container():
            price_col, remove_col = st.columns((1,1))
            with price_col:
                st.write(f"<center><h5>Price: ${product['price']}</h5></center>", unsafe_allow_html=True)
            with remove_col:  
                if st.button(f'Remove', key=f"remove_{product['name']}", use_container_width=True):
                    removeFromCart(product['id'])
                    st.session_state['prod_msg'] = f"{product['name']} has been removed from your cart!"
                    st_autorefresh(interval=1, limit=2)

if len(user_cart) == 0:
    st.write("<center><h5>Your cart is empty</h5></center>", unsafe_allow_html=True)

else:
    # Calculate total
    cart_total = sum([product['price'] for product in user_cart])

    with st.container():
        st.write("<hr>", unsafe_allow_html=True)
        st.write("<center><h5>Total: $"+str(round(cart_total, 2))+"</h5></center>", unsafe_allow_html=True)
        if st.button('Checkout', use_container_width=True):
            pass

# ---------------------------------------------------------------------
# End Body
# ---------------------------------------------------------------------