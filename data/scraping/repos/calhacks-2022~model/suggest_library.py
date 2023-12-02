import cohere
def get_recommendations(api_key, query):
    co = cohere.Client(api_key)
    response = co.generate(
    model='large',
    prompt='User: What is the best react library that I can install and use for making UI components in React\nList:@mui/material \nantd \nreactstrap \n@coreui/react \nsemantic-ui-react\n--\n'
        +'User: UI components\nList:\nsemantic-ui-react\nantd \nreactstrap\n@mui/material \n@coreui/react \n--\n'
        +'User: What is the best library that I can install and use for making API requests in React?\nList:axios\n@types/superagent\nky\npopsicle\nstream-http\n--\n'
        +'User: API requests\nList:axios\n@types/superagent\npopsicle\nstream-http\nky\n--\n'
        +'User: I want to include some icons\nList:material-design-icons \nfontawesome \nionicons \nreact-native-icons \nocticons \n--\n'
        +'User: icons\nList:octicons \nfontawesome \nionicons \nmaterial-design-icons \nreact-native-icons \n--\n'
        +'User: I want to install and parse XML to JSON in my React project for school\nList:xml2js \nxml2json \nsimple-xml-to-json \nfast-xml-parser\n@xmldom/xmldom\n--\nUser: XML to JSON\nList:xml2js \nfast-xml-parser\nxml2json \nsimple-xml-to-json \n@xmldom/xmldom\n--\n'
        'User: animation components\nList: reactstrap \nreact-spring \nreact-move \nremotion\nframer-motion\n--\n'
        +'User: I need to include animated components on my webpage\nList: react-move \nremotion\nframer-motion\nreactstrap \nreact-spring \n--\n'
        +'User: dropdown\nList:downshift\nreact-single-dropdown\nreact-select-search\nreact-dropdown-tree-select\nreact-multi-select-component\n--\n'
        +'User: I want to have a dropdown menu somewhere\nList:downshift\nreact-single-dropdown\nreact-multi-select-component\nreact-select-search\nreact-dropdown-tree-select\n--\n'
        +'User: data visualization\nList:react-chartjs-2\nrecharts\nvictory\n@visx/visx\nnivo\n--\n'
        +'User: I want to be able to visualize the data I have\nList:react-chartjs-2\n@visx/visx\nnivo\nrecharts\nvictory\n--\n'
        +'User: countdown timer\nList:react-timer-component\nreact-timer\nreact-countdown\n--\n'
        +'User: Best countdown timer integrations\nList:react-timer-component\nreact-timer\nreact-countdown\n--\n'
        +'User: validate email\nList:react-email-validator\nreact-js-validator\n@secretsofsume/valid-email\nvalidate-utility\nreact-weblineindia-email\n--\n'
        +'User: best way to validate emails in React\nList:@secretsofsume/valid-email\nvalidate-utility\nreact-js-validator\nreact-weblineindia-email\nreact-email-validator\n--\n'
        +'User: calendar integration\nList:react-calendar\nreact-big-calendar\n@fullcalendar/react\nreact-calendar-timeline\nexpo-calendar\n--\n'
        +'User: Best calendar integration for React\nList:@fullcalendar/react\nreact-calendar\nreact-big-calendar\nexpo-calendar\nreact-calendar-timeline\n--\n'
        +'User: authentication\nList:@auth0/auth0-react\n@azure/msal-react\nredux-auth-wrapper\n@axa-fr/react-oidc-context\nreact-aad-msal\n--\n'
        +'User: I want to authenticate the user\'s email input\nList: @axa-fr/react-oidc-context\n@azure/msal-react\nredux-auth-wrapper\nreact-aad-msal\n@auth0/auth0-react\n--\n'
        +'User: I want to add a video player to my React project. \nList: react-player \nreact-youtube \nvideo-react \n--\n'
        +'User: Video player\nList: react-player \nreact-youtube \nvideo-react \n--\n'
        +'User: I want a canvas component in React \nList: react-canvas-draw \nreact-canvas \nreact-sketch \ncreate-react-canvas \nreact-canvas-component \n--\n'
        +'User: canvas component\nList: react-canvas-draw \nreact-canvas \nreact-sketch \ncreate-react-canvas \nreact-canvas-component \n--\n'
        +'User: Add datepicker in React \nList: react-datepicker \nreact-datetime-picker \nreact-datetimepicker \nreact-fullcalendar \nreact-day-picker \nreact-picker \nreact-week-calendar \n--\n'
        +'User: datepicker\nList: \nreact-datetimepicker \nreact-fullcalendar \nreact-datepicker \nreact-day-picker \nreact-datetime-picker \nreact-picker \nreact-week-calendar \n--\n'
        +'User: I want a color picker component in react \nList: react-color \ncoloreact \nreact-input-color \nreact-color-picker \nreact-hex-picker \nreact-color-wheel \nreact-js-color \nreact-color-input \n--\n'
        +'User: Color picker\nList: react-color \nreact-color-wheel \ncoloreact \nreact-hex-picker \nreact-js-color \nreact-input-color \nreact-color-picker \nreact-color-input \n--\n'
        +'User: Loading bar for react UI \nList: react-load-more \nreact-flex-loader \nreact-custom-loader \nreact-redux-loading-bar \nreact-animated-loader \nreact-spinner \nreact-progress-bar \nreact-load-indicator \n--\n'
        +'User: Loading bar\nList: react-custom-loader \nreact-load-more \nreact-flex-loader \nreact-load-indicator \nreact-redux-loading-bar \nreact-spinner \nreact-animated-loader \nreact-progress-bar\n--\n'
        +'User: '
        + query
        + '\nList:',
    max_tokens=100,
    temperature=1,
    k=0,
    p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop_sequences=["--"],
    return_likelihoods='NONE')
    return response.generations[0].text.split("\n")[:-1]
